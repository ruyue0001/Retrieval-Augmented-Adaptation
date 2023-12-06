index_file_path = "roberta_data/gold_demo/wnut16/gold_demo_index.txt"
index_sent_path = "wnut16/train_sent.txt"
sent_path = "conll_03/train_sent.txt"
label_path = "conll_03/train_label_BIO_only.txt"
result_file_path = "roberta_data/gold_demo/wnut16/train.txt"

with open(index_file_path) as f:
    index = [line.strip() for line in f]
f.close()
print (len(index))

with open(index_sent_path) as f:
    index_sents = [line.strip() for line in f]
f.close()
print (len(index_sents))

with open(sent_path) as f:
    sents = [line.strip() for line in f]
f.close()
print (len(sents))

with open(label_path) as f:
    BIO_labels = [line.strip() for line in f]
f.close()
print (len(BIO_labels))

with open(result_file_path, "w") as r:
    for i in range(0, len(index)):
        index_i = index[i].split()
        sent_i = sents[i].split()
        label_i = BIO_labels[i].split()
        assert len(sent_i) == len(label_i)

        if index_i == []:
            assert len(sent_i) == 1
            assert label_i[0] == 'O'
            r.write(sent_i[0] + ' ' + label_i[0] + '\n\n')

        else:
            for j in range(0, len(sent_i)):
                r.write(sent_i[j] + ' ' + label_i[j] + '\n')
            r.write('<EOS> B-X\n')
            for k in index_i:
                retrieve_text = index_sents[int(k)].split()
                for token in retrieve_text:
                    r.write(token + ' ' + 'B-X' + '\n')

            r.write('\n')
r.close()