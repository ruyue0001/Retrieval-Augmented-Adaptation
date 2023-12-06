import json
import string
import collections

inf_result_path = "llama_inf_data/eval_result/ner_conll03_no_demo_lr3e-4_r16_alpha16_toi_aet_0/FIN_no_demo.txt"
gold_label_path = "FIN/test_label_token.txt"

with open(inf_result_path) as f:
    pred_labels = [line.strip() for line in f]
    print (len(pred_labels))
f.close()

with open(gold_label_path) as f:
    gold_labels = [line.strip() for line in f]
    print (len(gold_labels))
f.close()

assert len(pred_labels) == len(gold_labels)

gold_spans = []
pred_spans = []

def normalize_span(s):
    def repalce_punc(text):
        exclude = set(string.punctuation)
        for c in exclude:
            return text.replace(c, " ")

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(repalce_punc(lower(s)))


def calc_f1_score(gold_spans, pred_spans):
    # assert len(gold_spans) == len(pred_spans)
    common = collections.Counter(gold_spans) & collections.Counter(pred_spans)
    num_same = sum(common.values())
    if len(gold_spans) == 0 or len(pred_spans) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_spans == pred_spans)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_spans)
    recall = 1.0 * num_same / len(gold_spans)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1



for line in gold_labels:
    if line != '':
        spans = line.split('\t')
        for span in spans:
            gold_spans.append(normalize_span(span))
print ("Ground Truth span num:", len(gold_spans))

for line in pred_labels:
    assert line != ''
    if "None" in line:
        continue
    else:
        tmp = line.split(', ')
    for t in tmp:
        pred_spans.append(normalize_span(t))
print ("Pred span num:", len(pred_spans))

p, r, f = calc_f1_score(gold_spans, pred_spans)
print ("precision:", p, "recall:", r, "F1:", f)