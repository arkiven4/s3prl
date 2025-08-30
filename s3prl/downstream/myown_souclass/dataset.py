import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 20
EXAMPLE_DATASET_SIZE = 200


class SndClassifiDataset(Dataset):
    def __init__(self, split , **kwargs):
        self.root = kwargs['file_path']
        df = pd.read_csv(kwargs['meta_data'])
        #df['file_path'] = self.root + '/' + df['file_path']
        self.label_list = ['etc', 'speech', 'breathe', 
                    'cough', 'burp', 'gasp', 'sneeze', 
                    'sniffle', 'respiratory', 'throat-clearing', 
                    'vomit', 'silence', 'hiccup'] #list(df['label'].unique())
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=kwargs['train_dev_seed'])

        if split == 'test':
            self.table = test_df
        else:
            print(f'[SndClassifiDataset] - there are {len(train_df)} train and {len(test_df)} test found')
            print(f'[SndClassifiDataset] - label list {self.label_list}')
            self.table = train_df

        self.X = self.table['file_path'].tolist()   
        self.labels = self.table['label'].values

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        # assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def __getitem__(self, idx):
        wav = self._load_wav(self.X[idx])
        label = self.label_list.index(self.labels[idx])
        return wav, label

    def __len__(self):
        return len(self.X)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
