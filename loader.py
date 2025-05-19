import pandas as pd
import glob
import numpy as np
import json
import os

import torch
from torch.utils.data import Dataset

import whisper

class BirdDataset(Dataset):
    def __init__(self, audio_path, csv_path, subset, id_mapping_file, train_ratio=0.8):
        self.audio_path = audio_path
        self.df = pd.read_csv(csv_path)

        # preprocess the label to unique ID
        if not os.path.exists(f"./{id_mapping_file}.json"):
            labels = self.df['primary_label'].unique().tolist()

            self.id_tables = {}
            for idx, l in enumerate(labels):
                self.id_tables[l] = idx

            json_str = json.dumps(self.id_tables, indent=2)

            # 寫入檔案
            with open(f"./{id_mapping_file}.csv", "w") as f:
                f.write(json_str)
        else:
            f = open(f"./{id_mapping_file}.json", "r")
            self.id_tables = json.load(f)

        # split the subsets: train, valid
        idx_to_use = np.random.shuffle(np.arange(len(self.df)))
        assert subset in ['train', 'valid'], f"Invalid subset: {subset}"
        if subset == 'train':
            self.df = self.df.iloc[:int(len(self.df) * train_ratio)]
        else:
            self.df = self.df.iloc[int(len(self.df) * train_ratio):]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the audio file path and label
        target_row = self.df.iloc[idx]

        audio_file = target_row['filename']
        x = whisper.load_audio(os.path.join(self.audio_path, audio_file))
        x = whisper.pad_or_trim(x)

        y = target_row['primary_label']
        y = torch.LongTensor([self.id_tables[y]])
        
        return {"input_values": x, "labels": y}
