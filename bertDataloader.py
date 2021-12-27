import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')



map_labels = {
    'business':0,
    'entertainment':1,
    'sport':2,
    'tech':3,
    'politics':4
}

class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, df):
    self.labels = [map_labels[label] for label in df['category']]
    self.texts = [tokenizer(text, padding='max_length', max_length = 512, 
                            truncation=True, return_tensors="pt") for text in df['text']]
  

  def classes(self):
    return self.labels
  
  def __len__(self):
    return len(self.labels)
  
  def get_batch_labels(self, idx):
    return np.array(self.labels[idx])
  
  def get_batch_texts(self, idx):
    return self.texts[idx]

  def __getitem__(self, idx):
    return torch.from_numpy(self.get_batch_texts(idx)), torch.from_numpy(self.get_batch_labels(idx))

def getDataloader(df : pd.DataFrame) -> tuple:
    """
        This function returns a tuple : (train_dataloader, val_dataloader, test_dataloader)
    """
    train_val_df, test_df = train_test_split(df, test_size = 0.1, random_state =27)
    train_df, val_df = train_test_split(train_val_df, test_size = 0.12, random_state=27)

    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)
    test_dataset = CustomDataset(test_df)

    train_dataloader = DataLoader(train_dataset, batch_size = 64)
    val_dataloader = DataLoader(val_dataset, batch_size = 64)
    test_dataloader = DataLoader(test_dataset, batch_size = 64)
    return (train_dataloader, val_dataloader, test_dataloader)