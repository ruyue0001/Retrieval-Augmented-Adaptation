import json
#
# input_sent_path = "beauty/beauty.labeled.sent.txt"
# result_path = "llama_inf_data/BT/BT_no_demo.json"
#
# dataset_data = []
#
# with open(input_sent_path) as f:
#     for line in f:
#         tmp = {}
#         tmp["instruction"] = "Given the sentence, assign a sentiment label from ['positive', 'neutral', 'negative']. Return label only without any other text.\n\n"
#         tmp["input"] = line.strip()
#         dataset_data.append(tmp)
#
# print (dataset_data[0])
# print (dataset_data[1])
#
# with open(result_path, "w") as f:
#     json.dump(dataset_data, f)


index_path = "chatgpt_data/M-BT/M-BT_gold_demo_index.txt"
index_sent_path = "music/music.labeled.short.sent.txt"     #source
index_label_path = "music/music.labeled.short.label.txt"
input_sent_path = "beauty/beauty.labeled.sent.txt"    #target

result_path = "llama_inf_data/M-BT/M-BT_gold_demo.json"

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

with open(input_sent_path) as f:
    input_sents = [line.strip() for line in f]
    print (len(input_sents))
f.close()

# with open(output_label_token_path) as f:
#     output_label_tokens = [line.strip() for line in f]
#     print (len(output_label_tokens))
# f.close()

dataset_data = []

for i in range(0, len(index)):
    tmp = {}
    ins = "Given the sentence, assign a sentiment label from ['positive', 'neutral', 'negative']. Return label only without any other text.\n\n"
    idxs = index[i].split()
    for idx in idxs:
        sent = index_sents[int(idx)]
        label = index_labels[int(idx)]

        ins += "Text: " + sent + '\n'
        if label == '2':
            ins += "Label: positive\n"
        elif label == '1':
            ins += "Label: neutral\n"
        else:
            assert label == '0'
            ins += "Label: negative\n"


    tmp["instruction"] = ins
    tmp["input"] = input_sents[i]

    # output_label_token = output_label_tokens[i]
    # if output_label_token:
    #     tmp["output"] = ", ".join(output_label_token.split('\t'))
    # else:
    #     tmp["output"] = "None."
    #
    dataset_data.append(tmp)

print (dataset_data[0])
print (dataset_data[1])

with open(result_path, "w") as f:
    json.dump(dataset_data, f)


