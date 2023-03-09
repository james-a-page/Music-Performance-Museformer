import os
import pickle

import pandas as pd
import numpy as np

from remi_edit import REMI
from musicxmlannotations import genannotations
from miditok import Event
from utils import generate_tokens

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler


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


    def _build_tgt_bar_examples(self, tgt, BAR_IDX):
        tgt_bars = self._split_bars(tgt, BAR_IDX)

        prev_tgt_bars = []
        for idx, bar in enumerate(tgt_bars):
            prev_bars = []
            for prev_bar_idx in self.previous_bars:
                if idx - prev_bar_idx >= 0:
                    prev_bars.append(tgt_bars[idx - prev_bar_idx])

            # Flatten, add bars
            flattened = []
            if prev_bars != []:
                for i in prev_bars:  # TODO special token
                    flattened += [BAR_IDX]
                    flattened += i

                flattened += [BAR_IDX]
                    
            prev_tgt_bars.append(flattened)  # TODO flatten

        return prev_tgt_bars


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

            """
            print("Building dataset")
            metadata = pd.read_csv(self.metadata_path)
            metadata = metadata[['xml_score', 'midi_score', 'midi_performance']]
            metadata = metadata.drop(bad_xmls)

            # Build bar examples
            examples = {'src': [], 'prev_tgt': [], 'tgt': []}
            BAR_IDX = tokenizer.vocab._event_to_token['Bar_None']
            counter = 0
            for idx in range(len(metadata)):
                print(idx)
                xml, midi_src, midi_target = metadata.iloc[idx]

                instructions = genannotations.main(os.path.join(self.dataset_path, xml), time=True, verbose=False)
                src = generate_tokens(self.tokenizer, os.path.join(self.dataset_path, midi_src), instructions)
                tgt = generate_tokens(self.tokenizer, os.path.join(self.dataset_path, midi_target), instructions)

                prev_tgt = self._build_tgt_bar_examples(tgt, BAR_IDX)
                examples['src'] += src
                examples['prev_tgt'] += prev_tgt
                examples['tgt'] += tgt

                assert 1 == 0

            data = pd.DataFrame.from_dict(examples)
            data.to_csv(self.dataset_save_path, index=False)

            return data
            """


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

        assert len(src) < self.max_example_len
        assert len(tgt) < self.max_example_len

        if len(src) < len(tgt):  # Pad src with <SEP> tokens
            no_tokens = len(tgt) - len(src)
            src += [self.SEP_IDX] * no_tokens

        elif len(src) > len(tgt):  # Pad tgt with <EOS> tokens
            no_tokens = len(src) - len(tgt)
            tgt += [self.EOS_IDX] * no_tokens

        assert len(src) == len(tgt)

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
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), collate_fn=PadCollate(PAD_IDX))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices), collate_fn=PadCollate(PAD_IDX))

    return train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX
