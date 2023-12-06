import json
import string
import collections

# chat_gpt_result_path = "chatgpt_results/BC2GM_no_demo_output.json"
# gold_label_path = "BC2GM/test_label_token.txt"
# processed_gpt_result_path = "chatgpt_results/BC2GM_no_demo_output.txt"

chat_gpt_result_path = "llama_inf_data/eval_results/alpaca-lora-7b_inference/BT-BK_gold_demo.json"
gold_label_path = "book/book.labeled.label.txt"

# chat_gpt_result_path = "chatgpt_data/output/M_no_demo_output.json"
# gold_lael_path = "music/music.labeled.label.txt"


with open(gold_label_path) as f:
    label4sents = [line.strip() for line in f]
    print (len(label4sents))
f.close()

count = 0
count_positive_false = 0
count_neutral_false = 0
count_negative_false = 0

with open(chat_gpt_result_path) as f:
    eval_result = json.load(f)
    print (len(eval_result))
    assert len(label4sents) == len(eval_result)
    for i in range(0, len(eval_result)):
        output_i = eval_result[i]['output']
        output_i = output_i.lower()
        gold_label = label4sents[i]
        # assert output_i in ['positive', 'neutral', 'negative'], output_i
        if output_i.startswith('label: '):
            output_i = output_i[7:]
        if output_i.startswith('mixed/'):
            output_i = output_i[6:]
        if output_i not in ['positive', 'neutral', 'negative']:
            print (output_i)
        if gold_label == '2':
            if output_i == 'positive':
                count += 1
            else:
                count_positive_false += 1
                # print ("input:", eval_result[i]['input'])
                # print ("output:", output_i)
                # print ("gold: positive")

        if gold_label == '1':
            if output_i == 'neutral':
                count += 1
            else:
                count_neutral_false += 1
                # print ("input:", eval_result[i]['input'])
                # print ("output:", output_i)
                # print ("gold: neutral")

        if gold_label == '0':
            if output_i == 'negative':
                count += 1
            else:
                count_negative_false += 1
                # print ("input:", eval_result[i]['input'])
                # print ("output:", output_i)
                # print ("gold: negative")

    print ('Accuracy:', count/len(label4sents))
    print ('Pos false:', count_positive_false)
    print ('Neu false:', count_neutral_false)
    print ('neg false:', count_negative_false)

#     with open(chat_gpt_result_path) as f:
#         chat_gpt_result = json.load(f)
#         print (len(chat_gpt_result))
#         # first_output = chat_gpt_result[0]['output']
#         # print (first_output)
#         # if first_output.startswith("- "):
#         #     first_output = first_output.split('\n')
#         #     first_output = [i[2:] for i in first_output]
#         #     print (first_output)
#         for i in chat_gpt_result:
#             # print (i['output'])
#             output_i = i['output']
#             assert output_i != '', output_i
#             entities = []
#
#             if output_i.startswith ("- "):
#                 tmp = output_i.split('\n')
#                 for j in tmp:
#                     if j[2:].startswith("None"):
#                         continue
#                     elif j[2:].startswith("Entity"):
#                         entities.append(j[12:])
#                     else:
#                         if '(entity type' in j:
#                             indx = j.index('(entity type')
#                             entities.append(j[2:indx])
#                         else:
#                             entities.append(j[2:])
#
#             elif output_i.startswith("1. "):
#                 tmp = output_i.split('\n')
#                 for j in tmp:
#                     if j[3:].startswith("None"):
#                         continue
#                     else:
#                         if '(entity type' in j:
#                             indx = j.index('(entity type')
#                             entities.append(j[3:indx])
#                         else:
#                             entities.append(j[3:])
#
#             elif output_i.startswith("Entities:\n"):
#                 tmp = output_i[10:]
#                 if tmp.startswith("- "):
#                     tmp = tmp.split('\n')
#                     for j in tmp:
#                         if j[2:].startswith("None"):
#                             continue
#                         else:
#                             entities.append(j[2:])
#                 elif tmp.startswith("1. "):
#                     tmp = tmp.split('\n')
#                     for j in tmp:
#                         if j[3:].startswith("None"):
#                             continue
#                         else:
#                             entities.append(j[3:])
#
#             elif output_i.startswith("Entity:\n"):
#                 tmp = output_i[8:]
#                 if tmp.startswith("- "):
#                     tmp = tmp.split('\n')
#                     for j in tmp:
#                         if j[2:].startswith("None"):
#                             continue
#                         else:
#                             entities.append(j[2:])
#                 elif tmp.startswith("1. "):
#                     tmp = tmp.split('\n')
#                     for j in tmp:
#                         if j[3:].startswith("None"):
#                             continue
#                         else:
#                             entities.append(j[3:])
#
#             elif output_i.startswith("Entity:"):
#                 tmp = output_i[8:]
#                 tmp = tmp.split('\n')
#                 assert len(tmp) == 1
#                 entities.append(tmp[0])
#
#             elif output_i.startswith("Entity 1:"):
#                 tmp = output_i.split('\n')
#                 for j in tmp:
#                     if j[10:].startswith("None"):
#                         continue
#                     else:
#                         entities.append(j[10:])
#
#             elif output_i.startswith('None'):
#                 entities = []
#
#             else:
#                 if '\n' in output_i:
#                     tmp = output_i.split('\n')
#                 else:
#                     tmp = output_i.split(', ')
#                 for j in tmp:
#                     if j.startswith("None"):
#                         continue
#                     else:
#                         entities.append(j)
#
#
#             r.write("\t".join(entities) + '\n')
#             for s in entities:
#                 pred_spans.append(normalize_span(s))
#
# print (len(gold_spans), len(pred_spans))
# p, r, f = calc_f1_score(gold_spans, pred_spans)
# print ("precision:", p, "recall:", r, "F1:", f)