import os
import argparse

from dataset import load_data
from utils import set_seed, load_config, create_mask
from models.model.transformer import Transformer
from tqdm import tqdm
from model import Seq2SeqTransformer

import torch


def evaluate(cfg_file):
    cfg = load_config(cfg_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    # Set the random seed
    set_seed(seed=cfg["generic"].get("seed"))

    train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX = load_data(data_cfg=cfg["data"])

    #model = Transformer(cfg["transformer"], len(tokenizer.vocab._token_to_event), SOS_IDX, PAD_IDX, device).to(device)
    model = Seq2SeqTransformer(num_encoder_layers=2, num_decoder_layers=2, emb_size=512, nhead=8,
                               src_vocab_size=len(tokenizer.vocab._token_to_event), tgt_vocab_size=len(tokenizer.vocab._token_to_event),
                               dim_feedforward=512, dropout=0.0).to(device)
    model.load_state_dict(torch.load(cfg["eval"].get("load_path")))
    model.eval()

    # Test model on input
    decoded_tokens = [SOS_IDX]
    
    src, tgt = next(iter(train_loader))
    src = src.to(device).cuda()
    print(src)
    print(src.shape[1])

    input_tokens = torch.tensor([[SOS_IDX]]).cuda()
    with torch.no_grad():
        for i in tqdm(range(50)):
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, input_tokens, PAD_IDX, device)

            logits = model(src, input_tokens, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            logits = logits[:, logits.shape[1]-1, :]
            top_token = torch.argmax(logits).item()

            print(top_token)
            print(tgt[:, i+1])

            if top_token == EOS_IDX:
                break

            decoded_tokens.append(top_token)
            input_tokens = torch.cat((input_tokens, torch.tensor([[top_token]]).cuda()), dim=1)

    print(decoded_tokens)

    # Dump to midi
    src_midi = tokenizer.tokens_to_midi(src.tolist(), [(0, False)])
    src_midi.dump('src.mid')

    gen_midi = tokenizer.tokens_to_midi([decoded_tokens], [(0, False)])
    gen_midi.dump('gen.mid')

    tgt_midi = tokenizer.tokens_to_midi(tgt.tolist(), [(0, False)])
    tgt_midi.dump('tgt.mid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ttttt")
    parser.add_argument(
        "config",
        default="configs/asap.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    evaluate(cfg_file=args.config)
