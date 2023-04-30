from typing import List, Tuple, Dict
import collections
import random

import yaml

# from remi_edit import REMI
import numpy as np
from tqdm import tqdm
from preprocess import encoding_to_MIDI

import torch


Token_List = List[int]


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    path: path to YAML configuration file
    return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    return cfg


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(tgt, device, PAD_IDX):
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

    tgt_padding_mask = (tgt == PAD_IDX)
    return tgt_mask, tgt_padding_mask


def revert_example(example):
    out = []
    for b in example:
        new_b = []
        for n in b:
            if (n - 4) < 0:
                break

            new_b.append(n - 4)

        if len(new_b) == 8:
            out.append(new_b)

    return out


def greedy_decode(file_path, model, test_loader, PAD_IDX, SOS_IDX, EOS_IDX, device, save_src=False):
    """
    Decode single example from test dataloader greedily
    """
    tgt_input = [[[SOS_IDX, SOS_IDX, SOS_IDX, SOS_IDX, SOS_IDX, SOS_IDX, SOS_IDX, SOS_IDX]]]
    tgt_input = torch.tensor(tgt_input).to(device).cuda()

    # Get example
    batch = next(iter(test_loader))
    src, tgt, src_pad_mask, _ = batch
    
    src, tgt, src_pad_mask = src[0].unsqueeze(0).to(device).cuda(), tgt[0].unsqueeze(0).to(device).cuda(), \
                             src_pad_mask[0].unsqueeze(0).to(device).cuda()

    # TODO EOS
    with torch.no_grad():
        for i in tqdm(range(400)):
            tgt_pad_mask = torch.ones((1, tgt_input.shape[0]), dtype=torch.int64).to(device).cuda()

            outputs = model(src, tgt_input, src_pad_mask, tgt_pad_mask)

            # Decode
            decoded_example = []
            for out in outputs:
                final_out = out[:, out.shape[1]-1, :]
                top_token = torch.argmax(final_out).item()

                decoded_example.append(top_token)
                
            # Add decoded to next input
            decoded_example = torch.tensor(decoded_example).unsqueeze(-2).unsqueeze(-2).to(device).cuda()
            tgt_input = torch.concatenate((tgt_input, decoded_example), dim=1)

    # Dump to midi
    if save_src:
        src = revert_example(src.tolist()[0])
        midi = encoding_to_MIDI(src)
        midi.dump("{}_src.mid".format(file_path))

        tgt = revert_example(tgt.tolist()[0])
        midi = encoding_to_MIDI(tgt)
        midi.dump("{}_tgt.mid".format(file_path))
    
    decoded_tokens_list = tgt_input.detach().tolist()[0]
    decoded_tokens_list_rev = revert_example(decoded_tokens_list)
    if len(decoded_tokens_list_rev) == 0:
        return decoded_tokens_list
    try:
        midi = encoding_to_MIDI(decoded_tokens_list_rev)
        midi.dump("{}_gen.mid".format(file_path))
    except:
        print("Unparseable Generated Midi")
        return None
    return decoded_tokens_list
