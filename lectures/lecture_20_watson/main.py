# tf-idf vector
# sklearn/mintsim

# сеть с входом на 2 вектора
# выход - 3-х значный вектор (softmax function)

# torch/keras использовать для написания сетки

# import pandas as pd
# import torch.nn as nn
# import torch

# class Model(nn.Module):
#     def __init__(self) -> None:
#         super(torch.Model, self).__init__()
#         # super().__init__(*args, **kwargs)
#         self.sequence = [nn.Conv2d(1, 32, (3, 3)),
#                     nn.Conv2d(32, 64, (3, 3)),
#                     nn.MaxPool2d((2, 2), ),
#                     nn.Conv2d(64, 128, (3,3)) ,
#                     nn.Conv2d(128, 128, (3, 3)),
#                     nn.MaxPool2d((2,2)),
#                     nn.MaxPool2d((28, 28)),]
#         self.linear = nn.Linear(128, 26)
#         self.softmax = nn.Softmax()

#     def read_image(self, path):
#         self.train_df = pd.read(path+"/train.csv")
#         # self.test_df = pd.read(path+"/test.csv")

#     def forward(self, X):
#         res = self.sequence[0](X)
#         for i in self.sequence[1:]:
#             res = i(res)
#         res = self.linear(res.view(-1))
#         return self.softmax(res)

import torch
from torch import nn
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split="train")
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
