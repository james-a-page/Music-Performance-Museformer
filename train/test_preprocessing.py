import os
import argparse

from miditoolkit import MidiFile

import pandas as pd
from utils import load_config
from preprocess import MIDI_to_encoding
from tqdm import tqdm
import matplotlib.pyplot as plt


def evaluate(cfg_file):
    cfg = load_config(cfg_file)

    metadata_path = cfg["data"].get("metadata_path")
    dataset_path = cfg["data"].get("dataset_path")

    data = pd.read_csv(metadata_path)
    data = data[['midi_score']]

    all_bar_len = []
    for idx in tqdm(range(len(data))):
        midi_path = data.iloc[idx]['midi_score']
        midi = MidiFile(os.path.join(dataset_path, midi_path))

        tokens = MIDI_to_encoding(midi)
        tokens_flat = [j for i in tokens for j in i]
        all_bar_len.append(len(tokens_flat))

    plt.hist(all_bar_len)
    plt.savefig("dummy_name.png")


    #print("Estimated no. tokens: {}".format(max_token_idx))
    #print("Average example length: {}".format(total_tokens_len / len(data)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ttttt")
    parser.add_argument(
        "config",
        default="configs/asap.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    evaluate(cfg_file=args.config)
