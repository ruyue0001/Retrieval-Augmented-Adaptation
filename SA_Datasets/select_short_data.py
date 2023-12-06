process_file_path = 'beauty/beauty.labeled'

process_file_sent_path = process_file_path + ".short.sent.txt"
process_file_label_path = process_file_path + ".short.label.txt"

with open(process_file_path) as f:
    datasets = [line.strip() for line in f]
print (len(datasets))
f.close()

unlabeled = False

datasets_sents = []
datasets_labels = []

if unlabeled:
    for data in datasets:
        if len(data.split()) < 50:
            datasets_sents.append(data)

    print (len(datasets_sents))
    with open(process_file_sent_path, "w") as r:
        for sent in datasets_sents:
            r.write(sent + '\n')
    r.close()

else:
    for data in datasets:
        tmp = data.split('\t')
        if len(tmp) == 2 and len(tmp[1].split()) < 50:
            datasets_sents.append(tmp[1])
            datasets_labels.append(tmp[0])
    print (len(datasets_sents))
    print (len(datasets_labels))

    with open(process_file_sent_path, "w") as r:
        for sent in datasets_sents:
            r.write(sent + '\n')
    r.close()

    with open(process_file_label_path, "w") as r:
        for label in datasets_labels:
            r.write(label + '\n')
    r.close()


#labeled
#BT  2477
#E   2118
#BK  1627
#M   872

#unlabled
#BT  2523
#E   2347
#BK  2119
#M   1129