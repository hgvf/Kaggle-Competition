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
            pass

        with open(feats_key, 'r') as f:
            self.X_feats = f.readlines()[0].split(',')

        self.general_feats = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        self.label = 'label'

        self.X = self.data[self.X_feats + self.general_feats].to_numpy(dtype=np.float32)
        self.y = self.data[self.label].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        X_feats = self.X[idx]
        label = self.y[idx]

        x = torch.from_numpy(X_feats).to(torch.float32)
        y = torch.FloatTensor([label])
        
        return {"input": x, "label": y}