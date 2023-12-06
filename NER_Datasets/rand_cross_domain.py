query_sent_path = "bc5cdr/test_sent.txt"
# source_lable_path = "conll_03_english/train_label.txt"
index_sent_path = "conll_03/train_sent.txt"
# target_label_path = "wnut16/train_label.txt"
results_path = "chatgpt_data/bc5cdr_rand_demo_index.txt"

with open (query_sent_path) as f:
    query_sents = [line.strip() for line in f]
    print (len(query_sents))
f.close()

with open(index_sent_path) as f:
    index_sents = [line.strip() for line in f]
    print (len(index_sents))
f.close()

import random

with open(results_path, 'w') as f:
    for q_sent in query_sents:
        if q_sent == "-DOCSTART-":
            f.write('\n')
        else:
            results = random.sample(range(0, len(index_sents)), 5)
            for i in results:
                f.write(str(i) + ' ')
            f.write('\n')

f.close()