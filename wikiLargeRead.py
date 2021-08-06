import pandas as pd
import numpy as np
import re
pathToDataset = 'dataset\\'

df = pd.DataFrame()

with open(pathToDataset + 'wikiTrainSrc.txt', encoding='utf-8') as f:
    df['old'] = pd.Series(f.readlines())

with open(pathToDataset + 'wikiTrainDst.txt', encoding= 'utf-8') as f:
    df['new'] = pd.Series(f.readlines())

#remove all non alpha numeric characters
df['old'] = pd.Series([re.sub(r'[^a-zA-Z0-9_ ]', '',str(x)) for x in df['old']])
df['new'] = pd.Series([re.sub(r'[^a-zA-Z0-9_ ]', '',str(x)) for x in df['new']])



df.to_csv('train.csv', index=False)
