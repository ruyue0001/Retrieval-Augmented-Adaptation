import collections
import string

test_result_path = "taggers/gold_BioNLP09_xlmr-first_6epoch_2batch_2accumulate_0.000005lr_10000lrrate_eng_monolingual_crf_fast_norelearn_sentbatch_sentloss_finetune_nodev_conll-bionlp09_gold_demo_ner4/test.tsv"

process_result_path = "no_demo_BC2GM_test.txt"

with open(test_result_path) as f:
    test_results = [line.strip() for line in f]
# print (len(test_results))
f.close()

gold_spans = []
pred_spans = []
tokens = []
gold_labels = []
pred_labels = []

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

for line in test_results:
    line = line.split()
    if line != []:
        tokens.append(line[0])
        gold_labels.append(line[1])
        pred_labels.append(line[2])

# print (tokens[0],tokens[1],tokens[2],gold_labels[0], gold_labels[1],gold_labels[2])

def extract_spans(token_list, label_list):
    assert len(token_list) == len(label_list)
    spans = []
    tmp = []
    flag = False
    for i in range(0, len(label_list)):
        if label_list[i] == 'O':
            if flag:
                spans.append(normalize_span(' '.join(tmp)))
                tmp = []
                flag = False
            else:
                continue

        elif label_list[i] == 'B':
            if flag == True:
                spans.append(normalize_span(' '.join(tmp)))
                tmp = []
            tmp.append(token_list[i])
            flag = True
            if i == len(label_list) - 1:
                spans.append(normalize_span(' '.join(tmp)))
                tmp = []

        elif label_list[i] == 'I':
            assert  flag == True
            tmp.append(token_list[i])
            if i == len(label_list) - 1:
                spans.append(normalize_span(' '.join(tmp)))
                tmp = []

    return spans

gold_spans = extract_spans(tokens, gold_labels)
pred_spans = extract_spans(tokens, pred_labels)

# print (gold_spans[0], gold_spans[1], gold_spans[2])
# print (pred_spans[0], pred_spans[1], pred_spans[2])
# print (pred_spans[1])

p, r, f = calc_f1_score(gold_spans, pred_spans)
print ("precision:", p, "recall:", r, "F1:", f)

print (len(gold_spans), len(pred_spans))

# span = extract_spans(['New', 'Orleans', 'Mother'], ['B', 'I', 'I'])
# print (span)
print (gold_labels.count('B'), pred_labels.count('B'))