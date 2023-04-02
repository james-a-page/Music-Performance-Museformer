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

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


class ASAPDataset(Dataset):
    def __init__(self, data_cfg, tokenizer, SOS_IDX, EOS_IDX, PAD_IDX, SEP_IDX):
        self.dataset_path = data_cfg.get("dataset_path")
        self.metadata_path = data_cfg.get("metadata_path")
        self.new_tokens_dir = data_cfg.get("new_tokens_dir")
        self.dataset_save_path = data_cfg.get("dataset_save_path")
        self.max_example_len = data_cfg.get("max_example_len")

        self.SOS_IDX = SOS_IDX
        self.EOS_IDX = EOS_IDX
        self.PAD_IDX = PAD_IDX
        self.SEP_IDX = SEP_IDX

        self.tokenizer = tokenizer
        self.data = self._build_dataset(tokenizer)


    def _build_dataset(self, tokenizer):
        """
        Creates dataframe. Drops corrupted files, and examples that are too long. Constructs examples
        from specific bars
        """
        if os.path.exists(self.dataset_save_path):
            return pd.read_csv(self.dataset_save_path)
        else:
            print("Building dataset")
            data = pd.read_csv(self.metadata_path)
            data = data[['midi_score']]

            # Drop examples that are too long
            long_examples = []
            for idx in tqdm(range(len(data))):  # TODO store tokens in dataset
                midi_path = data.iloc[idx]['midi_score']
                midi = MidiFile(os.path.join(self.dataset_path, midi_path))

                tokens = tokenizer(midi)

                if len(tokens) + 2 > self.max_example_len:  # + 2 for SOS and EOS tokens
                    long_examples.append(idx)

            data = data.drop(long_examples).reset_index(drop=True)
            data.to_csv(self.dataset_save_path, index=False)

            print("Built")
            print('{:} long examples dropped, which was {:.2f} of the whole dataset'.format(len(long_examples), len(long_examples) / len(data)))

            return data


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        midi_path = self.data.iloc[idx]['midi_score']
        midi = MidiFile(os.path.join(self.dataset_path, midi_path))

        tokens = self.tokenizer(midi)
        tokens = [self.SOS_IDX] + tokens + [self.EOS_IDX]

        assert len(tokens) <= self.max_example_len

        return torch.tensor(tokens)


class PadCollate:
    def __init__(self, PAD_IDX):
        self.PAD_IDX = PAD_IDX

    def __call__(self, batch):
        max_seq_length = max([len(x) for x in batch])
        tokens = pad_sequence(batch, batch_first=True, padding_value=self.PAD_IDX)

        return tokens


def build_tokenizer(data_cfg: dict):
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 128}
    nb_velocities = 32
    additional_tokens = {
        'Chord': False,
        'Rest': False,
        'Tempo': True,
        'Program': False,
        'TimeSignature': True,
        'rest_range': (2, 32),  # (half, 8 beats)
        'nb_tempos': 512,  # nb of tempo bins
        'tempo_range': (1, 400)
    }  # (min, max)

    tokenizer = REMI(pitch_range,
                        beat_res,
                        nb_velocities,
                        additional_tokens,
                        mask=True,
                        pad=True,
                        sos_eos=True,
                        sep=True,)

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    SEP_IDX = 4

    return tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, SEP_IDX


def load_data(data_cfg: dict):
    tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, SEP_IDX = build_tokenizer(data_cfg)

    dataset = ASAPDataset(data_cfg, tokenizer, SOS_IDX, EOS_IDX, PAD_IDX, SEP_IDX)

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

    return train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX
