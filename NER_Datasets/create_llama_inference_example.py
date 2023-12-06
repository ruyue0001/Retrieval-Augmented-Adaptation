import json

input_sent_path = "FIN/test_sent.txt"
result_path = "llama_data/no_demo/FIN_no_demo.json"

dataset_data = []

with open(input_sent_path) as f:
    for line in f:
        tmp = {}
        tmp["instruction"] = "Please identify all entities from the input text. If there is no entity, please output None.\n"
        tmp["input"] = line.strip()
        dataset_data.append(tmp)

print (dataset_data[0])
print (dataset_data[1])

with open(result_path, "w") as f:
    json.dump(dataset_data, f)


# index_path = "chatgpt_data/gold_demo/FIN_gold_demo_index.txt"
# index_sent_path = "conll_03/train_sent.txt"
# index_label_token_path = "conll_03/train_label_token.txt"
# input_sent_path = "FIN/test_sent.txt"
# # output_label_token_path = "FIN/test_label_token.txt"
#
# result_path = "llama_data/gold_demo/FIN_gold_demo.json"
#
# with open(index_sent_path) as f:
#     index_sents = [line.strip() for line in f]
#     print (len(index_sents))
# f.close()
#
# with open(index_label_token_path) as f:
#     index_label_tokens = [line.strip() for line in f]
#     print (len(index_label_tokens))
# f.close()
#
# with open(index_path) as f:
#     index = [line.strip() for line in f]
#     print (len(index))
# f.close()
#
# with open(input_sent_path) as f:
#     input_sents = [line.strip() for line in f]
#     print (len(input_sents))
# f.close()
#
# # with open(output_label_token_path) as f:
# #     output_label_tokens = [line.strip() for line in f]
# #     print (len(output_label_tokens))
# # f.close()
#
# dataset_data = []
#
# for i in range(0, len(index)):
#     tmp = {}
#     ins = "Please identify all entities from the input text. If there is no entity, please output None.\n\n"
#     idxs = index[i].split()
#     for idx in idxs:
#         sent = index_sents[int(idx)]
#         Entity = index_label_tokens[int(idx)]
#         ins += "Sentence: " + sent + '\n'
#         if Entity:
#             ins += "Entity: " + Entity + '\n'
#         else:
#             ins += "Entity: None\n"
#
#     tmp["instruction"] = ins
#     tmp["input"] = input_sents[i]
#
#     # output_label_token = output_label_tokens[i]
#     # if output_label_token:
#     #     tmp["output"] = ", ".join(output_label_token.split('\t'))
#     # else:
#     #     tmp["output"] = "None."
#     #
#     dataset_data.append(tmp)
#
# print (dataset_data[0])
# print (dataset_data[1])
#
# with open(result_path, "w") as f:
#     json.dump(dataset_data, f)


