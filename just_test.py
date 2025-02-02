# import tokenizers

# tokenizer=tokenizers.BertWordPieceTokenizer("./model/RoBERTa/vocab.txt")
# tokenizer.enable_padding(length=16)
# tokenizer.enable_truncation(max_length=16)

# sentence = '河西区海河沿线的新房，均价30000，带装修，看看去吗，优惠点位很大，五一特惠'
# encode = tokenizer.encode(sentence, sentence)
# print(encode.ids)
# print(encode.type_ids)
# print(encode.tokens)
# print(encode.attention_mask)
###-------------------------------------###

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

df_test = pd.read_csv('./data/process_data/test.tsv',sep='\t',header=None)
df_test.columns = ['qid','rid','q','r']
df_test['label'] = 2
df_test[['qid','rid','label']].to_csv('submission_0.8019.tsv',index=False, header=None, sep='\t')

# #### multi label classify loss
# x = torch.rand(5, 3)
# y = torch.randint(0, 2, size=(5,))
# print(x, '\n', y)
# out_softmax = nn.functional.log_softmax(x)
# print(out_softmax)
# criteria = nn.NLLLoss()
# criteria2 = nn.CrossEntropyLoss()
# loss1 = criteria(out_softmax, y)
# loss2 = criteria2(x, y)
# loss3 = nn.functional.cross_entropy(x, y)
# print(loss1, loss2, loss3)

# #### binary classify loss
# np.random.seed(8)
# x = torch.from_numpy(np.random.randn(5,1)).view(5,)
# y = torch.from_numpy(np.ones((5,)))
# y2 = torch.from_numpy(np.zeros((5,1)))
# # print(x, '\n', y)
# print(x, '\n', y, '\n', y2)
# loss1 = nn.functional.binary_cross_entropy_with_logits(x, y.type_as(x))
# x_sigmoid = x.sigmoid()
# print(x_sigmoid)
# loss2 = nn.functional.binary_cross_entropy(x_sigmoid, y2)
# print(loss1)
# print(loss2)
# ####--------------------------------####

# x = torch.from_numpy(np.random.randn(5,1))
# print(x)
# x = x.sigmoid().numpy()
# print(x)
# x = [1 if i>0.5 else 0 for i in x]
# print(x)

