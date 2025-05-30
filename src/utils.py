import string
import regex as re

#################################
# Clarifying Question Utils
#################################

QA_INPUT = 'Question:'
QA_OUTPUT = 'Answer:'

ABSTAIN_OUTPUT = 'Abstain:'

CLARIFY_Q = 'Clarifying Question:'
CLARIFY_AS = 'Clarifying Answer Choices:'
CLARIFY_A = 'Clarifying Answer:'

def gen_clarify_q_prompt(
    qa_input,
    clarify_q=None,
):
    lines = []
    lines.append('{} {}'.format(QA_INPUT, qa_input.strip()))
    if clarify_q:
        lines.append('{} {}'.format(CLARIFY_Q, clarify_q.strip()))
        lines.append('')
    else:
        lines.append('{}'.format(CLARIFY_Q))

    return '\n'.join(lines)

def gen_clarify_a_prompt(
    qa_input,
    clarify_q,
    qa_output,
    clarify_a=None,
):
    lines = []
    lines.append('{} {}'.format(QA_INPUT, qa_input.strip()))
    lines.append('{} {}'.format(CLARIFY_Q, clarify_q.strip()))
    lines.append('{} {}'.format(QA_OUTPUT, qa_output.strip()))
    if clarify_a:
        lines.append('{} {}'.format(CLARIFY_A, clarify_a.strip()))
        lines.append('')
    else:
        lines.append('{}'.format(CLARIFY_A))
    return '\n'.join(lines)

def gen_direct_qa_output_prompt(
    qa_input,
    qa_output=None,
):
    lines = []
    lines.append('{} {}'.format(QA_INPUT, qa_input.strip()))
    if qa_output:
        lines.append('{} {}'.format(QA_OUTPUT, qa_output.strip()))
        lines.append('')
    else:
        lines.append('{}'.format(QA_OUTPUT))

    return '\n'.join(lines)

def gen_qa_output_prompt(
    qa_input,
    clarify_q,
    clarify_a,
    qa_output=None,
):
    lines = []
    lines.append('{} {}'.format(QA_INPUT, qa_input.strip()))
    lines.append('{} {}'.format(CLARIFY_Q, clarify_q.strip()))
    lines.append('{} {}'.format(CLARIFY_A, clarify_a.strip()))
    if qa_output:
        lines.append('{} {}'.format(QA_OUTPUT, qa_output.strip()))
        lines.append('')
    else:
        lines.append('{}'.format(QA_OUTPUT))

    return '\n'.join(lines)


#################################
# General Utils
#################################

def batched(iterable, n):
    return [iterable[batch_start:batch_start+n] for batch_start in range(0, len(iterable), n)]

def partitioned(iterable, n):
    partition_size = len(iterable) // n + bool(len(iterable) % n)
    partitions = batched(iterable, partition_size)
    assert len(partitions) == n
    return partitions

#################################
# QA Utils
#################################


def normalize(s):
    if s is None:
        return None
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ''.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em(answers1, answers2):
    if type(answers1) == str:
        answers1 = [answers1]
    if type(answers2) == str:
        answers2 = [answers2]
    if answers1 is None or answers2 is None:
        return False
    answers1 = set(normalize(a) for a in answers1 if a)
    answers2 = set(normalize(a) for a in answers2 if a)
    for a1 in answers1:
        for a2 in answers2:
            if a1 == a2:
                return True
    return False


def recall(pred_answers, gold_answers):
    pred_answers = {normalize(a) for a in pred_answers}
    gold_answers = {normalize(a) for a in gold_answers}
    return len(pred_answers.intersection(gold_answers))

def precision(pred_answers, gold_answers):
    pred_answers = {normalize(a) for a in pred_answers}
    gold_answers = {normalize(a) for a in gold_answers}
    return len(pred_answers.intersection(gold_answers))

def precision_recall(pred_answers, gold_answers):
    micro_rec = recall(pred_answers, gold_answers)
    micro_pre = precision(pred_answers, gold_answers)
    macro_rec = micro_rec / len(gold_answers)
    macro_pre = micro_pre / len(pred_answers)
    return {
        'macro_f1': 2 / (1 / macro_rec + 1 / macro_pre) if macro_rec and macro_pre else 0,
        'macro_rec': macro_rec,
        'micro_rec': micro_rec,
        'micro_pre': micro_pre,
        'micro_pred_total': len(pred_answers),
        'micro_gold_total': len(gold_answers),
    }
