import os
import re
import json
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from accelerate import Accelerator, PartialState

from utils import batched, partitioned

def make_chat(messages):
    chat = []
    for i, message in enumerate(messages):
        if i % 2:
            role = 'assistant'
        else:
            role = 'user'
        chat.append({
            "role": role,
            "content": message
        })
    return chat


def main(args):
    accelerator = Accelerator()
    distributed_state = PartialState()

    # OUTPUT PATH SETUP
    output_dir = os.path.split(args.input_path)[0]
    output_name = os.path.split(args.input_path)[1][:-len('jsonl')]
    output_name += f'{args.model}_rewards'
    if args.shard_total:
        output_name += f'.{args.shard_idx}of{args.shard_total}'
    output_name += '.jsonl'
    print(output_name)

    ## Loading Reward Model
    if args.model == 'starling':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.starling_path,
            quantization_config=bnb_config,
            device_map={'': distributed_state.process_index}
        )
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.truncation_side = "left"
    elif args.model == 'openassist':
        model = AutoModelForSequenceClassification.from_pretrained(
            'OpenAssistant/reward-model-deberta-v3-large-v2',
            device_map={'': distributed_state.process_index}
        )
        tokenizer = AutoTokenizer.from_pretrained('OpenAssistant/reward-model-deberta-v3-large-v2')
    else:
        raise ValueError
    model.eval().requires_grad_(False)

    ## Loading Input Data
    with open(args.input_path, 'r') as f:
        data = [json.loads(l) for l in f][:args.test]
    if args.shard_total:
        data = partitioned(data, args.shard_total)[args.shard_idx-1]
    data = partitioned(
        data,
        distributed_state.num_processes
    )[distributed_state.process_index]

    # Generating Prompts
    all_clarifications = []
    all_chats = []
    for ex in data:
        ex['dpo'] = {
            'clarifications': [],
            'has_best': True,
        }
        if ex['pred']['clarification']:
            ex['dpo']['clarifications'].append(ex['pred']['clarification'])
        ex['dpo']['clarifications'].extend(ex['pred']['clarification_samples'])

        for cl in ex['dpo']['clarifications']:
            all_clarifications.append(cl)
            all_chats.append(make_chat([ex['question'], cl['question']]))


    # Running Reward Models Prompts
    all_rewards = []
    all_batched_chats = batched(all_chats, args.batch_size)
    for batch_chats in tqdm(
        all_batched_chats,
        position=distributed_state.process_index,
        total=len(all_batched_chats)
    ):
        if args.model == 'starling':
            batch_chats_tokenized =  [
                tokenizer.apply_chat_template(chat, tokenize=False) for chat in batch_chats
            ]
            batch_inputs = tokenizer(
                batch_chats_tokenized,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            ).to(model.device)
            pass
        elif args.model == 'openassist':
            batch_questions = [chat[-2]['content'] for chat in batch_chats]
            batch_answers = [chat[-1]['content'] for chat in batch_chats]
            batch_inputs = tokenizer(
                batch_questions,
                batch_answers,
                padding=True,
                return_tensors='pt'
            ).to(model.device)
        else:
            raise ValueError
        outputs = model(**batch_inputs)
        scores = outputs.logits[:, 0].cpu().detach().tolist()
        all_rewards.extend(scores)

    for cl, reward in zip(all_clarifications, all_rewards):
        cl['reward'] = reward
    for ex in data:
        ranks = sorted(
            range(len(ex['dpo']['clarifications'])),
            key=lambda x: ex['dpo']['clarifications'][x]['reward'],
            reverse=True
        )
        for rank, cl_idx in enumerate(ranks):
            ex['dpo']['clarifications'][cl_idx]['rank'] = rank+1



    tmp_output_dir = os.path.join(output_dir, 'tmp')
    os.makedirs(tmp_output_dir, exist_ok=True)
    tmp_output_path = os.path.join(
        tmp_output_dir,
        f'{distributed_state.process_index}_{output_name}'
    )
    with open(tmp_output_path, 'w') as f:
        for ex in data:
            f.write(json.dumps(ex) + '\n')

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        all_outputs = []
        for process_idx in range(accelerator.num_processes):
            tmp_output_path = os.path.join(
                tmp_output_dir,
                f'{process_idx}_{output_name}'
            )
            with open(tmp_output_path, 'r') as f:
                all_outputs.extend(json.loads(l) for l in f)

        with open(os.path.join(output_dir, output_name), 'w') as f:
            for ex in all_outputs:
                f.write(json.dumps(ex) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--model')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test', type=int, default=None)
    parser.add_argument('--shard_idx', type=int, default=None)
    parser.add_argument('--shard_total', type=int, default=None)

    parser.add_argument(
        '--starling_path',
        default='/var/local/mjqzhang/cautious_rlhf/base_model/starling-rm'
    )

    cli_args = parser.parse_args()
    main(cli_args)
