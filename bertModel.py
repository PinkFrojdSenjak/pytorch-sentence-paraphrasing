from torchvision import transforms
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
  def __init__(self, dropout = 0.5, n_embeddings = 768, n_classes = 5):
    super(BertClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(n_embeddings, n_classes)
    self.relu = nn.ReLU()
    
  def forward(self, input_id, mask):
    _, pooled_output = self.bert(input_ids = input_id, attention_mask = mask, return_dict = False)
    dropout_output = self.drouput(pooled_output)
    linear_output = self.linear(dropout_output)
    result = self.relu(linear_output)
    return result
