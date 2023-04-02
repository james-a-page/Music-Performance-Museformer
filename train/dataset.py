import os
import pickle

import pandas as pd
import numpy as np

from remi_edit import REMI
from musicxmlannotations import genannotations
from miditok import Event
from utils import generate_tokens
from tqdm import tqdm
from miditoolkit import MidiFile
from preprocess import MIDI_to_encoding

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


class ASAPDataset(Dataset):
    def __init__(self, data_cfg, SOS_IDX, EOS_IDX, PAD_IDX):
        self.dataset_path = data_cfg.get("dataset_path")
        self.metadata_path = data_cfg.get("metadata_path")
        self.new_tokens_dir = data_cfg.get("new_tokens_dir")
        self.dataset_save_path = data_cfg.get("dataset_save_path")
        self.max_example_len = data_cfg.get("max_example_len")

        self.SOS_IDX = SOS_IDX
        self.EOS_IDX = EOS_IDX
        self.PAD_IDX = PAD_IDX

        # Build data
        self.data = pd.read_csv(self.metadata_path)[['midi_score', 'midi_performance']]
        self.data.to_csv(self.dataset_save_path, index=False)


    def __len__(self):
        return len(self.data)


    def _construct_and_shift(self, tokens):
        output = []
        output += [[0, 0, 0, 0, 0, 0, 0, 0]]  # SOS_IDX

        for bar in tokens:
            bar_shifted = [i + 3 for i in bar]
            output += [bar_shifted]

        output += [[1, 1, 1, 1, 1, 1, 1, 1]]  # EOS_IDX
        return output


    def __getitem__(self, idx):
        src_path, tgt_path = self.data.iloc[idx]

        src = MIDI_to_encoding(MidiFile(os.path.join(self.dataset_path, src_path)))
        src = self._construct_and_shift(src)

        tgt = MIDI_to_encoding(MidiFile(os.path.join(self.dataset_path, tgt_path)))
        tgt = self._construct_and_shift(tgt)

        return torch.tensor(src), torch.tensor(tgt)


class PadCollate:
    def __init__(self, PAD_IDX):
        self.PAD_IDX = PAD_IDX

    def __call__(self, batch):
        src, tgt = zip(*batch)
        
        # TODO fix padding, create separate list to track PAD tokens for both

        src = pad_sequence(src, batch_first=True, padding_value=self.PAD_IDX)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=self.PAD_IDX)

        # Construct padding masks
        src_pad_mask = src.ne(self.PAD_IDX).float()
        src_pad_mask = torch.sum(src_pad_mask, dim=-1)
        src_pad_mask = src_pad_mask > 0

        tgt_pad_mask = tgt.ne(self.PAD_IDX).float()
        tgt_pad_mask = torch.sum(tgt_pad_mask, dim=-1)
        tgt_pad_mask = tgt_pad_mask > 0

        return src, tgt, src_pad_mask, tgt_pad_mask


def load_data(data_cfg: dict):
    SOS_IDX = 0
    EOS_IDX = 1
    PAD_IDX = 2

    dataset = ASAPDataset(data_cfg, SOS_IDX, EOS_IDX, PAD_IDX)

    # Create splits
    indices = list(range(len(dataset)))
    if data_cfg.get("shuffle"):
        np.random.shuffle(indices)

    train_prop, val_prop, test_prop = data_cfg.get("dataset_split")
    train_split = int(np.floor(train_prop * len(dataset)))
    val_split = train_split + int(np.floor(val_prop * len(dataset)))
    train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]

    batch_size = data_cfg.get("batch_size")

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), collate_fn=PadCollate(PAD_IDX))
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(val_indices), collate_fn=PadCollate(PAD_IDX))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(test_indices), collate_fn=PadCollate(PAD_IDX))

    return train_loader, val_loader, test_loader, PAD_IDX, SOS_IDX, EOS_IDX
