import json
from collections import Counter
import argparse
from utils import precision_recall, normalize
from utils import em


def eval_respond(data, answers_key):
    greedy_metrics_counter = Counter()
    sample_metrics_counter = Counter()
    for ex in data:
        gold_answers = set(ex[answers_key])
        greedy_preds = [ex['pred']['response']]
        sample_preds = [r for r, _ in Counter([normalize(s) for s in ex['pred']['response_samples']]).most_common(len(gold_answers))]
        greedy_metrics_counter.update(precision_recall(greedy_preds, gold_answers))
        sample_metrics_counter.update(precision_recall(sample_preds, gold_answers))
        greedy_metrics_counter['any_em'] += em(greedy_preds, gold_answers)
        sample_metrics_counter['any_em'] += em(sample_preds, gold_answers)
    greedy_metrics = {
        'rec': greedy_metrics_counter['macro_rec'] / len(data),
        'f1': greedy_metrics_counter['macro_f1'] / len(data),
        'em': greedy_metrics_counter['any_em'] / len(data),
    }
    sample_metrics = {
        'rec': sample_metrics_counter['micro_rec'] / len(data),
        'f1': sample_metrics_counter['macro_f1'] / len(data),
        'em': sample_metrics_counter['any_em'] / len(data),
    }
    print('Greedy Metrics:')
    # print(json.dumps(greedy_metrics, indent=2))
    keys, vals = zip(*greedy_metrics.items())
    print('\t'.join(keys))
    print('\t'.join(['{:.3f}'.format(v) for v in vals]))

    print('Sample Metrics:')
    # print(json.dumps(sample_metrics, indent=2))
    keys, vals = zip(*sample_metrics.items())
    print('\t'.join(keys))
    print('\t'.join(['{:.3f}'.format(v) for v in vals]))

def eval_clarify(data, answers_key):
    metrics_counter = Counter()
    for ex in data:
        em_count = 0
        gold_answer_count = 0
        for answer in ex['pred']['clarification']['eval_answers']:
            if not answer[answers_key]:
                continue
            em_count += bool(
                em(answer['response'], answer['gold_response']) and
                answer['answer'] and
                (normalize(answer['gold_response']) not in normalize(answer['answer'])
                    or normalize(answer['gold_response']) not in normalize(ex['pred']['clarification']['question']))
            )
            gold_answer_count += 1
        metrics_counter['macro_f1'] += em_count / gold_answer_count
        metrics_counter['micro_f1'] += em_count
        metrics_counter['micro_total'] += gold_answer_count
        metrics_counter['any_em'] += bool(em_count)
    metrics = {
        'f1': metrics_counter['macro_f1'] / len(data),
        'em': metrics_counter['any_em'] / len(data),
    }
    keys, vals = zip(*metrics.items())
    print('\t'.join(keys))
    print('\t'.join(['{:.3f}'.format(v) for v in vals]))

    

def main(args):
    with open(args.input_path) as f:
        data = [json.loads(l) for l in f]
    ambig_data = [ex for ex in data if ex['isambig']]

    if args.mode == 'respond':
        print('NQ-Open Evaluations:')
        eval_respond(data, 'nq_answers')
        print()
        print('AmbigQA Evaluations:')
        eval_respond(ambig_data, 'answers')
    elif args.mode == 'clarify':
        print('NQ-Open Evaluations:')
        eval_clarify(data, 'is_nq')
        print()
        print('AmbigQA Evaluations:')
        eval_clarify(ambig_data, 'is_ambig')
    else:
        raise ValueError(args.mode)
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--mode')
    cli_args = parser.parse_args()
    main(cli_args)
