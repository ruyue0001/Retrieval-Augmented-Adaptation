# process_file_path = "music/music.train"
#
# process_file_sent_path = process_file_path + ".sent.txt"
# process_file_label_path = process_file_path + ".label.txt"
#
# with open(process_file_path) as f:
#     datasets = [line.strip() for line in f]
# print (len(datasets))
# f.close()
#
# datasets_sents = []
# datasets_labels = []
# for data in datasets:
#     tmp = data.split('\t')
#     if len(tmp) == 2:
#         datasets_sents.append(tmp[1])
#         datasets_labels.append(tmp[0])
# print (len(datasets_sents))
# print (len(datasets_labels))
#
# with open(process_file_sent_path, "w") as r:
#     for sent in datasets_sents:
#         r.write(sent + '\n')
# r.close()
#
# with open(process_file_label_path, "w") as r:
#     for label in datasets_labels:
#         r.write(label + '\n')
# r.close()

#beauty  2477
#ele  2118
#book 1627
#music 872


#chatgpt inference
#instruction
#demonstration: source labeled
#input: target test
#target test as the query
#source train/labeled as the index, serve as the labeled demonstration
query_sent_path = "beauty/beauty.labeled.sent.txt"  #target
index_sent_path = "music/music.labeled.short.sent.txt"   #source
results_path = "chatgpt_data/M-BT/M-BT_gold_demo_index.txt"

#roberta train
#source train <eos> target unlabeled
#source train as the query
#target unlabeled as the index


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