import json

# train without demonstrations
# train with source domain  - conll03
'''
input_sent_path = "conll_03/train_sent.txt"
input_label_path = "conll_03/train_label_token.txt"
result_path = "llama_train_data/no_demo/train_no_demo_500.json"

with open(input_sent_path) as f:
    input_sents = [line.strip() for line in f]
print (len(input_sents))
f.close()

with open(input_label_path) as f:
    input_labels = [line.strip() for line in f]
print (len(input_labels))
f.close()

assert len(input_sents) == len(input_labels)


def create_no_demo(input_sents, input_labels):
    dataset_data = []

    for i in range(0, 500):
        tmp = {}
        tmp["instruction"] = "Please identify all entities from the input text. If there is no entity, please output None.\n"
        tmp["input"] = input_sents[i]
        if input_labels[i] != '':
            tmp["output"] = ', '.join(input_labels[i].split('\t'))
        else:
            tmp["output"] = "None"
        dataset_data.append(tmp)

    return dataset_data

dataset_data = create_no_demo(input_sents, input_labels)

print (dataset_data[0])
print (dataset_data[1])
print (len(dataset_data))

with open(result_path, "w") as f:
    json.dump(dataset_data, f)
'''

# train with target demonstration, rand or gold
# source - conll-03
# target - FIN, wnut16, wnut17, BC2GM, BioNLP09, bc5cdr
# index_path : retrieved target sentence id
# index_sent_path : target unlabeled sentence
# input_sent_path: source sentence
# input_label_path : source label

index_path = "roberta_data/gold_demo/bc5cdr/gold_demo_index.txt"
index_sent_path = "bc5cdr/train_sent.txt"
input_sent_path = "conll_03/train_sent.txt"
input_label_path = "conll_03/train_label_token.txt"
# output_label_token_path = "FIN/test_label_token.txt"

result_path = "llama_train_data/gold_demo/bc5cdr_gold_demo.json"

with open(index_sent_path) as f:
    index_sents = [line.strip() for line in f]
    print (len(index_sents))
f.close()

with open(index_path) as f:
    index = [line.strip() for line in f]
    print (len(index))
f.close()

with open(input_sent_path) as f:
    input_sents = [line.strip() for line in f]
    print (len(input_sents))
f.close()

with open(input_label_path) as f:
    input_labels = [line.strip() for line in f]
print (len(input_labels))
f.close()

assert len(input_sents) == len(input_labels) == len(index)

dataset_data = []

for i in range(0, len(index)):
    tmp = {}
    ins = "Please identify all entities from the input text. If there is no entity, please output None.\n\n"
    idxs = index[i].split()
    for idx in idxs:
        sent = index_sents[int(idx)]
        # Entity = index_label_tokens[int(idx)]
        if len(sent.split()) < 50:
            ins += "Sentence: " + sent + '\n'
        else:
            print (sent)
        # if Entity:
        #     ins += "Entity: " + Entity + '\n'
        # else:
        #     ins += "Entity: None\n"

    tmp["instruction"] = ins
    tmp["input"] = input_sents[i]
    if input_labels[i] != '':
        tmp["output"] = ', '.join(input_labels[i].split('\t'))
    else:
        tmp["output"] = "None"

    dataset_data.append(tmp)

print (dataset_data[0])
print (dataset_data[1])
print (len(dataset_data))

with open(result_path, "w") as f:
    json.dump(dataset_data, f)