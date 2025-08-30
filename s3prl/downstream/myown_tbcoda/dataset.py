import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class TBCodaDataset(Dataset):
    def __init__(self, split , **kwargs):
        self.root = kwargs['file_path']
        df = pd.read_csv(kwargs['meta_data'] + "." + split)
        self.table = df
        print(f'[TBCodaDataset] - there are {len(self.table)} in {split} sets')

        self.label_list = list(self.table[kwargs['target_column']].unique())
        self.X = self.table['file_path'].tolist()   
        self.labels = self.table[kwargs['target_column']].values

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        # assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def __getitem__(self, idx):
        wav = self._load_wav(self.X[idx])
        label = self.labels[idx]
        return wav, label

    def __len__(self):
        return len(self.X)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
