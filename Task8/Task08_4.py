from tokenize import tokenize

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext import data
import tqdm

# tokenize = lambda x: x.split()
# TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True,
# fix_length=200)
# LABEL = data.Field(sequential=False, use_vocab=False)

def get_dataset(csv_data, text_field, label_field, test=False):
    fields = [("id", None), # we won't be needing the id, so we pass in None as the field
              ("comment_text", text_field), ("toxic", label_field)]
    examples = []
    if test:
    # 如果为测试集，则不加载label
        for text in tqdm(csv_data['comment_text']):
            examples.append(data.Example.fromlist([None, text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['comment_text'],csv_data['toxic'])):
                examples.append(data.Example.fromlist([None, text, label], fields))
    return examples, fields


train_data = pd.read_csv('train_toxic_comments.csv')
valid_data = pd.read_csv('valid_toxic_comments.csv')
test_data = pd.read_csv("test_toxic_comments.csv")
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)
# 得到构建Dataset所需的examples和fields
train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
valid_examples, valid_fields = get_dataset(valid_data, TEXT, LABEL)
test_examples, test_fields = get_dataset(test_data, TEXT, None, test=True)
# 构建Dataset数据集
train = data.Dataset(train_examples, train_fields)
valid = data.Dataset(valid_examples, valid_fields)
test = data.Dataset(test_examples, test_fields)


# 检查keys是否正确
print(train[0].__dict__.keys())
print(test[0].__dict__.keys())
# 抽查内容是否正确
print(train[0].comment_text)
