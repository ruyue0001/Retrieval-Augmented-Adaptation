query_sent_path = "conll_03/train_sent.txt"
# source_lable_path = "conll_03_english/train_label.txt"
index_sent_path = "BC2GM/train_sent.txt"
# target_label_path = "wnut16/train_label.txt"
results_path = "roberta_data/gold_demo/BC2GM/gold_demo_index.txt"

# import bert_score
# from bert_score import score
# import heapq
from simcse import SimCSE
model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

with open (query_sent_path) as f:
    query_sents = [line.strip() for line in f]
    print (len(query_sents))
f.close()

with open(index_sent_path) as f:
    index_sents = [line.strip() for line in f]
    print (len(index_sents))
f.close()
# #
# print (source_sents[1])
# print (target_sents[0])
# target_sents = target_sents[:50]
#
# for s_sent in source_sents:
#     scores = []
#     if s_sent == "-DOCSTART-":
#         continue
#     for t_sent in target_sents:
#         #(P, R, F1), hashcode = score([s_sent], [t_sent], lang='en', rescale_with_baseline=True, return_hash=True)
#         P, R, F1 = score([s_sent], [t_sent], lang='en', rescale_with_baseline=True, device="cuda:0")
#         scores.append(F1.item())
#         # print (scores)
#         # print (hashcode)
#     max_5_number = heapq.nlargest(5, scores)
#     max_5_index = list(map(scores.index, heapq.nlargest(5,scores)))
#     # print (max_5_number)
#     # print (max_5_index)
#     for max_index in max_5_index:
#         print (target_sents[max_index])

# p,R,F1 = score([source_sents[1]], [target_sents[0]], lang='en', rescale_with_baseline=True,device="cuda:0")
# print (F1.item())

model.build_index(index_sents)

# for q_sent in query_sents:
#     if q_sent == "-DOCSTART-":
#
#     results = model.search(q_sent, threshold=0)
#     index = []
#     for i in results:
#         index.append(target_sents.index(i[0]))
#         if i[1] > 0.5:
#             print(s_sent, i)
#     #print(index)

with open(results_path, 'w') as f:
    for q_sent in query_sents:
        if q_sent == "-DOCSTART-":
            f.write('\n')
        else:
            results = model.search(q_sent, threshold=0)
            index = []
            for i in results:
                index.append(index_sents.index(i[0]))
                # if i[1] > 0.5:
                #     print (query_sents.index(q_sent),q_sent, i)
            # print (index)
            for i in index:
                f.write(str(i)+ ' ')
            f.write('\n')

f.close()