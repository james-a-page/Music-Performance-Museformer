import os
import argparse

from dataset import load_data
from utils import set_seed, load_config

import torch


def dump_dataset(cfg_file):
    cfg = load_config(cfg_file)
    set_seed(seed=cfg["generic"].get("seed"))

    train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX = load_data(
        data_cfg=cfg["data"])
    
    split = ['train','val','test']

    for i, loader in enumerate([train_loader, val_loader, test_loader]):
        dump_tokens(split[i], loader)

    dump_vocab_dict(tokenizer)

    print(f'EOS: {EOS_IDX} \n SOS: {SOS_IDX} \n PAD: {PAD_IDX} \n ')


def dump_tokens(split, data):
    pass

def dump_vocab_dict(tokenizer):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ttttt")
    parser.add_argument(
        "config",
        default="configs/asap.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )

    args = parser.parse_args()

    dump_dataset(cfg_file=args.config)
