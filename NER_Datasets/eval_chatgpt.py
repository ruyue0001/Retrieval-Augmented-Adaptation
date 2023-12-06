import json
import string
import collections

# chat_gpt_result_path = "chatgpt_results/no_demo/bc5cdr_no_demo_output.json"
# gold_label_path = "bc5cdr/test_label_token.txt"
# processed_gpt_result_path = "chatgpt_results/bc5cdr_no_demo_output.txt"

chat_gpt_result_path = "llama_data/eval_result/alpaca-lora-7b_inference/BC2GM_rand_demo.json"
gold_label_path = "BC2GM/test_label_token.txt"
processed_gpt_result_path = "llama_data/eval_result/alpaca-lora-7b_inference/BC2GM_rand_demo_output_span.txt"



with open(gold_label_path) as f:
    label4sents = [line.strip() for line in f]
    print (len(label4sents))
f.close()

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

def normalize_sent(s):
    def repalce_punc(text):
        exclude = set(string.punctuation)
        for c in exclude:
            return text.replace(c, " ")


    def white_space_fix(text):
        return text.replace(" ","")


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



for label4sent in label4sents:
    if label4sent != '':
        label4sent = label4sent.split('\t')
        for label in label4sent:
            gold_spans.append(normalize_span(label))
print (len(gold_spans))

#check
gold_label_BIO_path = "BC2GM/test_label_BIO_only.txt"
with open(gold_label_BIO_path) as f:
    labelBIO = [line.strip() for line in f]
    print (len(labelBIO))
f.close()
count = 0
for i in range(0, len(labelBIO)):
    count_B = labelBIO[i].count('B')
    if label4sents[i] != '':
        count_entity = len(label4sents[i].split('\t'))
    else:
        count_entity = 0
    #print (i, count_B, count_entity, label4sents[i])
    assert count_B == count_entity
    count += count_B
print ("Entity num:", count)


with open (processed_gpt_result_path, "w") as r:
    with open(chat_gpt_result_path) as f:
        chat_gpt_result = json.load(f)
        print (len(chat_gpt_result))
        # first_output = chat_gpt_result[0]['output']
        # print (first_output)
        # if first_output.startswith("- "):
        #     first_output = first_output.split('\n')
        #     first_output = [i[2:] for i in first_output]
        #     print (first_output)
        for i in chat_gpt_result:
            # print (i['output'])
            output = i['output']
            assert output != '', output
            entities = []
            if 'input' in i.keys():
                input = i['input']
                if normalize_sent(output) == normalize_sent(input):
                    # print (input)
                    # print (output)
                    r.write('\n')
                    continue

            if "None" in output:
                r.write('\n')
                continue

            if "###" in output:
                indx = output.index('###')
                output_i = output[:indx]
                # print (output, output_i)
            else:
                output_i = output


            if output_i.startswith ("- "):
                tmp = output_i.split('\n')
                for j in tmp:
                    if j[2:].startswith("Entity"):
                        entities.append(j[12:])
                    else:
                        if '(entity type' in j:
                            indx = j.index('(entity type')
                            entities.append(j[2:indx])
                        else:
                            entities.append(j[2:])

            elif output_i.startswith("1. "):
                tmp = output_i.split('\n')
                for j in tmp:
                    if '(entity type' in j:
                        indx = j.index('(entity type')
                        entities.append(j[3:indx])
                    else:
                        entities.append(j[3:])

            elif output_i.startswith("Entities:\n"):
                tmp = output_i[10:]
                if tmp.startswith("- "):
                    tmp = tmp.split('\n')
                    for j in tmp:
                        entities.append(j[2:])
                elif tmp.startswith("1. "):
                    tmp = tmp.split('\n')
                    for j in tmp:
                        entities.append(j[3:])

            elif output_i.startswith("Entity:\n"):
                tmp = output_i[8:]
                # print (output_i, tmp)
                if tmp.startswith("- "):
                    tmp = tmp.split('\n')
                    for j in tmp:
                        entities.append(j[2:])
                elif tmp.startswith("1. "):
                    tmp = tmp.split('\n')
                    for j in tmp:
                        entities.append(j[3:])

            elif output_i.startswith("Entity: \n"):
                tmp = output_i[9:]
                # print (output_i, tmp)
                if tmp.startswith("- "):
                    tmp = tmp.split('\n')
                    for j in tmp:
                        entities.append(j[2:])
                elif tmp.startswith("1. "):
                    tmp = tmp.split('\n')
                    for j in tmp:
                        entities.append(j[3:])

            elif output_i.startswith("Entity:"):
                tmp = output_i[8:]
                # print (output_i, tmp)
                if '\n' in output_i:
                    tmp = tmp.split('\n')
                else:
                    tmp = tmp.split(', ')
                for j in tmp:
                    entities.append(j)


            elif output_i.startswith("Entity 1:"):
                if '\n' in output_i:
                    tmp = output_i.split('\n')
                else:
                    tmp = output_i.split(', ')
                for j in tmp:
                    if j != '':
                        entities.append(j[10:])


            else:
                if '\n' in output_i:
                    tmp = output_i.split('\n')
                else:
                    tmp = output_i.split(', ')
                for j in tmp:
                    entities.append(j)


            r.write("\t".join(entities) + '\n')
            for s in entities:
                pred_spans.append(normalize_span(s))

print (len(gold_spans), len(pred_spans))
p, r, f = calc_f1_score(gold_spans, pred_spans)
print ("precision:", p, "recall:", r, "F1:", f)