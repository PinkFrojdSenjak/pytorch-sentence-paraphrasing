from bertDataloader import getDataloader
from bertModel import BertClassifier
from torchvision import transforms
import pandas as pd
import torch
import numpy as np
from torch import nn
import pandas as pd
import torch
import numpy as np

def train(dataloader, model, loss_fn, optimizer, device):
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f'Batch {batch}/{num_batches}: Loss: {loss}')
        total_loss+=loss
    return total_loss / num_batches

def test(dataloader, model, loss_fn, optimizer, device):
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    model.eval()
    total_loss, acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = loss_fn(pred, y)
            acc+= (pred.argmax(1) == y).type(torch.float).sum().item()
    total_loss/=num_batches
    acc/=num_samples
    print(f'Loss: {total_loss}  Accuracy: {acc}')
    return total_loss, acc

df = pd.read_csv('../drive/MyDrive/tempFolder/bbc-text.csv')
train_dataloader, val_dataloader, test_dataloader = getDataloader(df)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
model = BertClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
