import os
import argparse
import shutil

from dataset import load_data
from utils import set_seed, load_config, greedy_decode
from model import EncoderRNN, DecoderRNN

import torch
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def train(cfg_file):
    cfg = load_config(cfg_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    log_dir = cfg["generic"].get("log_dir")

    """
    if cfg['generic'].get('clear_log') and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    """
    writer = SummaryWriter(log_dir=log_dir)

    # Set the random seed
    set_seed(seed=cfg["generic"].get("seed"))

    train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX = load_data(data_cfg=cfg["data"])

    encoder = EncoderRNN(cfg["encoder"], len(tokenizer.vocab._token_to_event), device).to(device)
    decoder = DecoderRNN(cfg["decoder"], len(tokenizer.vocab._token_to_event), device).to(device)

    if device == "cuda":
        print_gpu_utilization()

    # Optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=float(cfg["encoder"].get("lr")))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=float(cfg["decoder"].get("lr")))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    epochs = cfg["training"].get("epochs")
    batch_size = cfg["data"].get("batch_size")

    # Save stuff
    save = cfg["training"].get("save")
    save_every_x_epochs = cfg["training"].get("save_every_x_epochs")
    save_dir = cfg["training"].get("save_dir")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print(len(train_loader))

    for epoch in range(epochs):
        print(epoch)
        encoder.train()
        decoder.train()
        for batch_idx, batch in enumerate(train_loader):
            tokens, tokens_pred = batch

            this_batch_size = tokens.shape[0]
            input_length = tokens.shape[1]
            target_length = tokens_pred.shape[1]

            loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Encode input
            encoder_hidden = encoder(tokens.to(device).cuda())

            decoder_input = torch.full((this_batch_size, 1), SOS_IDX, device=device).cuda()
            decoder_hidden = encoder_hidden.cuda()

            # Decoding, teacher forcing
            cum_loss = 0
            for idx in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                decoder_output = decoder_output[:, 0, :]
                decoder_input = tokens_pred[:, idx].unsqueeze(1).cuda()

                l = criterion(decoder_output.cpu(), tokens_pred[:, idx])
                loss += l
                cum_loss += l

            if device == "cuda":
                print_gpu_utilization()
                
            avg_loss = cum_loss / target_length

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            # Tensorboard
            global_step = epoch*len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', avg_loss, global_step)

        # Greedily decode last example
        decoded_tokens = greedy_decode(encoder, decoder, tokens[0], EOS_IDX, SOS_IDX, device)
 
        # Val
        val_loss = 0
        encoder.eval()
        decoder.eval()
        for batch_idx, batch in enumerate(val_loader):
            tokens, tokens_pred = batch

            this_batch_size = tokens.shape[0]
            input_length = tokens.shape[1]
            target_length = tokens_pred.shape[1]

            # Encode input
            with torch.no_grad():
                encoder_hidden = encoder(tokens.to(device).cuda())

                decoder_input = torch.full((this_batch_size, 1), SOS_IDX, device=device).cuda()
                decoder_hidden = encoder_hidden.cuda()

                # Decoding, teacher forcing
                cum_loss = 0
                for idx in range(target_length):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    decoder_output = decoder_output[:, 0, :]
                    decoder_input = tokens_pred[:, idx].unsqueeze(1).cuda()

                    l = criterion(decoder_output.cpu(), tokens_pred[:, idx])
                    cum_loss += l
                val_loss += cum_loss / target_length

                if device == "cuda":
                    print_gpu_utilization()

        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, global_step)

        # Save
        if (epoch + 1) % save_every_x_epochs == 0:
            torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder_{:}.pt".format(epoch)))
            torch.save(decoder.state_dict(), os.path.join(save_dir, "decoder_{:}.pt".format(epoch)))


if __name__ == "__main__":
    print("a")
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
    train(cfg_file=args.config)
