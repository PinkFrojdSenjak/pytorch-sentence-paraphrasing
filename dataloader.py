import torch
import numpy as np
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('valid.csv')

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# seeing the distribution of 
'''
seq_len = [len(str(i).split()) for i in train_df['old']]
pd.Series(seq_len).plot.hist(bins = 30).plot()
plt.show()
'''
#tokenizing
tokens_train = tokenizer.batch_encode_plus(
    train_df['old'].tolist(),
    max_length = 32,
    pad_to_max_length=True,
    truncation=True
)
