from bertDataloader import getDataloader
from bertModel import BertClassifier
from torchvision import transforms
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
import pandas as pd
import torch
import numpy as np

df = pd.read_csv('../drive/MyDrive/tempFolder/bbc-text.csv')

#train_dataloader, val_dataloader, test_dataloader = getDataloader(df)
print(torch.cuda.is_available())

pass