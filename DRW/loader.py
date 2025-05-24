# utf-8

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class DRWDataset(Dataset):
    def __init__(self, data_path, subset, train_ratio=0.8, feats_key='./useful_feats.txt'):
        self.data = pd.read_parquet(data_path)

        idx = int(len(self.data) * train_ratio)
        if subset == 'train':
            self.data = self.data.iloc[:idx]
        elif subset == 'val':
            self.data = self.data.iloc[idx:]
        else:
            raise ValueError("subset must be either 'train' or 'val'")

        with open(feats_key, 'r') as f:
            self.X_feats = f.readlines()[0].split(',')

        self.general_feats = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        self.label = 'label'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X_feats = self.data[self.X_feats].iloc[idx].to_numpy()
        general_feats = self.data[self.general_feats].iloc[idx].to_numpy()
        label = self.data[self.label].iloc[idx]

        x = torch.from_numpy(np.concatenate([X_feats, general_feats])).to(torch.float32)
        y = torch.FloatTensor([label])
        
        return {"input": x, "label": y}

data = DRWDataset(data_path='train.parquet', subset='train', train_ratio=0.8, feats_key='./useful_feats.txt')

for d in data:
    print(d['input'].shape)
    break