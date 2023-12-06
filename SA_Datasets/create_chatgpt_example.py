# file_path = "music/music.labeled.sent.txt"
# result_path = "chatgpt_data/M/M_no_demo.txt"
# with open (result_path, "w") as r:
#     with open(file_path) as f:
#         for line in f:
#             r.write("Given the sentence, assign a sentiment label from ['positive', 'neutral', 'negative']. Return label only without any other text.\n\n")
#             r.write("Text: " + line)
#             r.write("Label:" + "\n\n")
#             r.write("\n")

index_path = "chatgpt_data/BK-E/BK-E_rand_demo_index.txt"
index_sent_path = "book/book.labeled.short.sent.txt"   #source
index_label_path = "book/book.labeled.short.label.txt"
query_sent_path = "electronics/electronics.labeled.sent.txt"  #target
result_path = "chatgpt_data/BK-E/BK-E_rand_demo.txt"


with open(index_sent_path) as f:
    index_sents = [line.strip() for line in f]
    print (len(index_sents))
f.close()

with open(index_label_path) as f:
    index_labels = [line.strip() for line in f]
    print (len(index_labels))
f.close()

with open(index_path) as f:
    index = [line.strip() for line in f]
    print (len(index))
f.close()

with open(query_sent_path) as f:
    query_sents = [line.strip() for line in f]
    print (len(query_sents))
f.close()

with open(result_path, "w") as r:
    for i in range(0, len(index)):
        r.write("Given the sentence, assign a sentiment label from ['positive', 'neutral', 'negative']. Return label only without any other text.\n\n")
        idxs = index[i].split()
        for idx in idxs:
            sent = index_sents[int(idx)]
            label = index_labels[int(idx)]
            r.write("Text: " + sent + '\n')
            if label == '2':
                r.write("Label: positive\n")
            elif label == '1':
                r.write("Label: neutral\n")
            else:
                assert  label == '0'
                r.write("Label: negative\n")
        r.write('\n')
        r.write("Text: " + query_sents[i] + '\n')
        r.write("Label:\n\n")
        r.write("\n")