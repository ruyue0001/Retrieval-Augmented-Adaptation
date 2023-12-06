query_sent_path = "beauty/beauty.labeled.sent.txt"     #target
# source_lable_path = "conll_03_english/train_label.txt"
index_sent_path = "music/music.labeled.short.sent.txt"    #source
# target_label_path = "wnut16/train_label.txt"
results_path = "chatgpt_data/M-BT/M-BT_rand_demo_index.txt"

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
        results = random.sample(range(0, len(index_sents)), 5)
        for i in results:
            f.write(str(i) + ' ')
        f.write('\n')

f.close()