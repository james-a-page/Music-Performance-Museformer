import os
import argparse
import numpy as np
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

    print(f'EOS: {EOS_IDX} \n SOS: {SOS_IDX} \n PAD: {PAD_IDX} \n')

# Museformer assumes bar tokens at the end of the bar => add a bar token before the end of sequence token.
def dump_tokens(split, data):
    score_tok, perf_tok = [],[]
    print(f"Loading {split} tokens...")
    for c,batch in enumerate(iter(data)):
        print(f"{c}/{len(iter(data))}")
        score, perf = batch[0], batch[1]
        if len(score) != len(perf):
            print("Batch length mismatch!")
            continue
        else:
            for i in range(len(score)):
                score_temp = score[i]
                perf_temp = perf[i]
                score_temp = np.insert(score_temp.numpy(), -1, 4)
                perf_temp = np.insert(perf_temp.numpy(), -1, 4)
                score_tok.append(" ".join(str(x) for x in list(score_temp)))
                perf_tok.append(" ".join(str(x) for x in list(perf_temp)))
    print(f"Done")
    print(f"Writing {split}.score...")
    with open(f'data/tokens/{split}.score', 'w') as score_f:
        score_f.writelines(line + '\n' for line in score_tok)
    print(f"Done")
    print(f"Writing {split}.perf...")
    with open(f'data/tokens/{split}.perf', 'w') as perf_f:
        perf_f.writelines(line + '\n' for line in perf_tok)
    print(f"Done")

def dump_vocab_dict(tokenizer):
    vocab = tokenizer.vocab
    with open('./museformer/data/meta/our_dict.txt', 'w') as dict_f:
        for tok in list(vocab.event_to_token.values()):
            if tok not in [0,1,2]:
                dict_f.write(str(tok) + ' 1\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        default="configs/asap.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )

    args = parser.parse_args()
    dump_dataset(cfg_file=args.config)
