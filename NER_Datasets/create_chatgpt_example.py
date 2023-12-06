file_path = "FIN/test_sent.txt"
result_path = "chatgpt_data/no_demo/FIN_no_demo.txt"
with open (result_path, "w") as r:
    with open(file_path) as f:
        for line in f:
            r.write("Please identify all entities from the given text. If there is no entity, please output None.\n\n")
            r.write("Text: " + line)
            r.write("Entity:" + "\n\n")
            r.write("\n")

# index_path = "chatgpt_data/rand_demo/FIN_rand_demo_index.txt"
# index_sent_path = "conll_03/train_sent.txt"
# index_label_token_path = "conll_03/train_label_token.txt"
# file_path = "FIN/test_sent.txt"
# result_path = "chatgpt_data/rand_demo/FIN_rand_demo.txt"
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
# with open(file_path) as f:
#     sents = [line.strip() for line in f]
#     print (len(sents))
# f.close()
#
# with open(result_path, "w") as r:
#     for i in range(0, len(index)):
#         r.write("Please identify all entities from the given text. If there is no entity, please output None.\n\n")
#         idxs = index[i].split()
#         for idx in idxs:
#             sent = index_sents[int(idx)]
#             Entity = index_label_tokens[int(idx)]
#             r.write("Text: " + sent + '\n')
#             if Entity:
#                 r.write("Entity: " + Entity + '\n\n')
#             else:
#                 r.write("Entity: None\n\n")
#         r.write("Text: " + sents[i] + '\n')
#         r.write("Entity:\n\n")
#         r.write("\n")