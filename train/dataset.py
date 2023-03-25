import os
import pickle

import pandas as pd
import numpy as np

from remi_edit import REMI
from musicxmlannotations import genannotations
from miditok import Event
from utils import generate_tokens
from tqdm import tqdm

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

        # Build tokenizer, get bad rows
        self.tokenizer, bad_xmls = self._build_tokenizer(tokenizer)

        # Build data
        self.data = self._build_dataset(tokenizer, bad_xmls)


    def _split_bars(self, example, BAR_IDX):
        bars = []
        current_bar = []
        for i in example:
            if i == BAR_IDX and current_bar != []:
                bars.append(current_bar)
                current_bar = []
            elif i != BAR_IDX:
                current_bar.append(i)

        return bars


    def _get_bar_similarity(self, b1, b2):
        pass


    def _get_top_similarities(self, src_bar_seq, tgt_bar):
        pass


    def _build_dataset_bars(self, tokenizer, bad_xmls):
        """
        Input, all robotic bars + idxs of most similar bars to tgt
        Target, single tgt bar
        """
        if os.path.exists(self.dataset_save_path):
            return pd.read_csv(self.dataset_save_path)
        else:
            print("Building dataset")
            data = pd.read_csv(self.metadata_path)
            data = data[['xml_score', 'midi_score', 'midi_performance']]
            data = data.drop(bad_xmls).reset_index(drop=True)

            BAR_IDX = tokenizer.vocab._event_to_token['Bar_None']

            examples = {'src': [], 'tgt_bar': []}
            for idx in tqdm(range(len(data))):
                if len(examples['src']) >= 20:
                    break

                xml, midi, midi_y = data.iloc[idx]

                instructions = genannotations.main(os.path.join(self.dataset_path, xml), time=True, verbose=False)
                src = generate_tokens(self.tokenizer, os.path.join(self.dataset_path, midi), instructions)
                tgt = generate_tokens(self.tokenizer, os.path.join(self.dataset_path, midi_y), instructions)

                if len(src) + 2 > self.max_example_len or len(tgt) + 2 > self.max_example_len:  # + 2 for SOS and EOS tokens
                    continue

                src_bars = self._split_bars(src, BAR_IDX)
                tgt_bars = self._split_bars(tgt, BAR_IDX)

                # Construct example for every target bar
                for b in tgt_bars:
                    examples['src'] += [src]
                    examples['tgt_bar'] += [b]

            data = pd.DataFrame.from_dict(examples)
            data.to_csv(self.dataset_save_path, index=False)
            return data


    def _build_dataset(self, tokenizer, bad_xmls):
        """
        Creates dataframe. Drops corrupted files, and examples that are too long. Constructs examples
        from specific bars
        """
        if os.path.exists(self.dataset_save_path):
            return pd.read_csv(self.dataset_save_path)
        else:
            print("Building dataset")
            data = pd.read_csv(self.metadata_path)
            data = data[['xml_score', 'midi_score', 'midi_performance']]
            data = data.drop(bad_xmls).reset_index(drop=True)

            # Drop examples that are too long
            long_examples = []
            for idx in range(len(data)):
                xml, midi, midi_y = data.iloc[idx]

                instructions = genannotations.main(os.path.join(self.dataset_path, xml), time=True, verbose=False)
                tokens = generate_tokens(self.tokenizer, os.path.join(self.dataset_path, midi), instructions)
                target_tokens = generate_tokens(self.tokenizer, os.path.join(self.dataset_path, midi_y), instructions)

                if len(tokens) + 2 > self.max_example_len or len(target_tokens) + 2 > self.max_example_len:  # + 2 for SOS and EOS tokens
                    long_examples.append(idx)

            data = data.drop(long_examples).reset_index(drop=True)
            data.to_csv(self.dataset_save_path, index=False)

            print("Built")
            print('{:} long examples dropped, which was {:.2f} of the whole dataset'.format(len(long_examples), len(long_examples) / len(data)))

            return data


    def _load_tokenizer(self, tokenizer):
        with open(os.path.join(self.new_tokens_dir, "new_tokens.pickle"), "rb") as f:
                new_instructions = pickle.load(f)

        for i in new_instructions:
            tokenizer.vocab.add_event(Event("Instruction", i, time=0, desc="Instruct_"+i))

        with open(os.path.join(self.new_tokens_dir, "bad_xmls.pickle"), "rb") as f:
            bad_xmls = pickle.load(f)

        return tokenizer, bad_xmls


    def _build_tokenizer(self, tokenizer):
        """
        Adds all instruction tokens, found in any example
        """
        # Load from previous generation
        if os.path.exists(self.new_tokens_dir):
            return self._load_tokenizer(tokenizer)
        else:
            print("Building tokenizer")
            os.mkdir(self.new_tokens_dir)

            all_xml = pd.read_csv(self.metadata_path)['xml_score'].tolist()

            bad_xmls = list()
            new_instructions = list()
            for idx, i in enumerate(all_xml):
                path = os.path.join(self.dataset_path, i)
                try:
                    instructions = genannotations.main(path, time=True, verbose=False)
                except (KeyError, IndexError) as _:
                    bad_xmls.append(idx)
                    continue

                # If we have instructions, add to tokenizer vocab
                for _, j in instructions:
                    if j not in new_instructions and j != '':
                        tokenizer.vocab.add_event(Event("Instruction", j, time=0, desc="Instruct_"+j))
                        new_instructions.append(j)

            # Dump to file
            with open(os.path.join(self.new_tokens_dir, "new_tokens.pickle"), "wb+") as f:
                pickle.dump(new_instructions, f)

            with open(os.path.join(self.new_tokens_dir, "bad_xmls.pickle"), "wb+") as f:
                pickle.dump(bad_xmls, f)

            print("Built")
            print('{:} bad examples dropped, which was {:.2f} of the whole dataset'.format(len(bad_xmls), len(bad_xmls) / len(all_xml)))
            
            return tokenizer, bad_xmls


    def _buildtokenizer_whitelist(self, tokenizer):
        """
        Adds instruction tokens found within a whitelist
        """

        return tokenizer


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        xml, midi, midi_y = self.data.iloc[idx]

        instructions = genannotations.main(os.path.join(self.dataset_path, xml), time=True, verbose=False)
        src = generate_tokens(self.tokenizer, os.path.join(self.dataset_path, midi), instructions)
        src = [self.SOS_IDX] + src + [self.EOS_IDX]

        tgt = generate_tokens(self.tokenizer, os.path.join(self.dataset_path, midi_y), instructions)
        tgt = [self.SOS_IDX] + tgt + [self.EOS_IDX]

        assert len(src) <= self.max_example_len
        assert len(tgt) <= self.max_example_len

        return torch.tensor(src), torch.tensor(tgt)


class PadCollate:
    def __init__(self, PAD_IDX):
        self.PAD_IDX = PAD_IDX

    def __call__(self, batch):
        tokens, target_tokens = zip(*batch)
        
        max_seq_length = max([len(x) for x in tokens])

        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.PAD_IDX)
        target_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=self.PAD_IDX)

        return tokens, target_tokens


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
