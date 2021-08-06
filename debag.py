# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import numpy as np
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split


# %%
df = pd.read_csv('train.csv')
df =df.dropna()


# %%
train_old, temp_old, train_new, temp_new = train_test_split(df['old'], df['new'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3)
val_old, test_old, val_new, test_new = train_test_split(temp_old, temp_new, 
                                                                random_state=2018, 
                                                                test_size=0.5) 


# %%
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda")


# %%
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# %%
oldTrainTokens = tokenizer.batch_encode_plus(
    train_old.to_list(),
    max_length = 32,
    padding = 'max_length',
    truncation=True
)
newTrainTokens = tokenizer.batch_encode_plus(
    train_new.to_list(),
    max_length = 32,
    padding = 'max_length',
    truncation=True
)
oldValTokens = tokenizer.batch_encode_plus(
    val_old.to_list(),
    max_length = 32,
    padding = 'max_length',
    truncation=True
)
newValTokens = tokenizer.batch_encode_plus(
    val_new.to_list(),
    max_length = 32,
    padding = 'max_length',
    truncation=True
)

oldTestTokens = tokenizer.batch_encode_plus(
    test_old.to_list(),
    max_length = 32,
    padding = 'max_length',
    truncation=True
)
newTestTokens = tokenizer.batch_encode_plus(
    test_new.to_list(),
    max_length = 32,
    padding = 'max_length',
    truncation=True
)


# %%
totensor = torch.LongTensor(oldTrainTokens['input_ids'])
tntensor = torch.LongTensor(newTrainTokens['input_ids'])

tomask = torch.LongTensor(oldTrainTokens['attention_mask'])
tnmask = torch.LongTensor(newTrainTokens['attention_mask'])

votensor = torch.LongTensor(oldValTokens['input_ids'])
vntensor = torch.LongTensor(newValTokens['input_ids'])

vomask = torch.LongTensor(oldValTokens['attention_mask'])
vnmask = torch.LongTensor(newValTokens['attention_mask'])

testotensor = torch.LongTensor(oldTestTokens['input_ids'])
testntensor = torch.LongTensor(newTestTokens['input_ids'])

testomask = torch.LongTensor(oldTestTokens['attention_mask'])
testnmask = torch.LongTensor(newTestTokens['attention_mask'])


# %%
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
batch_size = 32
trainData = TensorDataset(totensor, tntensor, tomask, tnmask)
valData = TensorDataset(votensor, vntensor, vomask, vnmask)
testData = TensorDataset(testotensor, testntensor, testomask, testnmask)


# %%
trainSampler = RandomSampler(trainData)
valSampler = RandomSampler(valData)
testSampler = RandomSampler(testData)


# %%
trainDataloader = DataLoader(trainData, sampler=trainSampler, batch_size=batch_size)
valDataloader = DataLoader(valData, sampler=valSampler, batch_size=batch_size)
trainDataloader = DataLoader(testData, sampler=testSampler, batch_size=batch_size)


# %%
from TransformerModel import Transformer
model = Transformer(num_heads = 8,num_encoder_layers = 6, num_decoder_layers = 6, dropout_p = 0.1 ,dim_model = 512, num_tokens = 32 )
model = model.to(device = device)

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr = 1e-5)
crossEntropy = nn.NLLLoss()
epochs = 10


# %%
def train():
  
    model.train()

    total_loss = 0
  
  # empty list to save model predictions
    total_preds=[]
  
  # iterate over batches
    for step,batch in enumerate(trainDataloader):
        if step % 50 == 0 and not step == 0:
          print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(trainDataloader)))

    # push the batch to gpu
        batch = [r.to(device) for r in batch]
        
        old_id, new_id, old_mask, new_mask = batch
        #old_id = old_id[:,:-1]
        #new_id = new_id[:,1:]
        print(old_id.size(), new_id.size())
        model.zero_grad()
        tgt_mask = model.get_tgt_maska(size = 32)
        
        output = model(src = old_id,tgt =  new_id,tgt_mask = tgt_mask,src_pad_mask =  old_mask,tgt_pad_mask = new_mask)
       
        new_id = new_id.to(torch.float32)
        print(output[0], new_id[0])
        print('Dim: ', output.size(),new_id.size(), output.dtype, new_id.dtype )
        
        break
        loss = crossEntropy(new_id, output)
        total_loss+=loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    avg_loss = total_loss / len(trainDataloader)
    return avg_loss
        


# %%
train()


# %%
from TransformerModel import PositionalEncoding
tempSrc = totensor[:32]
tempTgt = tntensor[:32]

emb = nn.Embedding(100000, 512)
dvasrc = emb(tempSrc)
dvatgt = emb(tempTgt)
triSrc = (dvasrc)
triSrc


