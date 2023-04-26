import os
import argparse
import shutil

from dataset import load_data
from utils import set_seed, load_config, greedy_decode, create_mask
from models.model.transformer import Transformer
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter


def objective(discriminator, output, target, kl_divergence):
    discrimination_error = discriminator(output, target)
    N = 60000.

    return discrimination_error + kl_divergence / N


def train(cfg_file):
    cfg = load_config(cfg_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    log_dir = cfg["generic"].get("log_dir")
    if cfg['generic'].get('clear_log') and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Set the random seed
    set_seed(seed=cfg["generic"].get("seed"))

    train_loader, val_loader, test_loader, PAD_IDX, SOS_IDX, EOS_IDX = load_data(data_cfg=cfg["data"], pretrain=cfg["training"].get("pretrain"))

    model = Transformer(cfg["transformer"], cfg["training"].get('pretrain'), SOS_IDX, PAD_IDX, device).to(device)
    if cfg["training"].get("load"):
        model.load_state_dict(torch.load(cfg["eval"].get("load_path")))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["transformer"].get("lr")), betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    epochs = cfg["training"].get("epochs")

    # Save stuff
    save = cfg["training"].get("save")
    save_every = cfg["training"].get("save_every")
    save_dir = cfg["training"].get("save_dir")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    decode = cfg["training"].get("decode")
    decode_every = cfg["training"].get("decode_every")
    decode_dir = cfg["training"].get("decode_dir")
    if not os.path.exists(decode_dir):
        os.mkdir(decode_dir)

    for epoch in range(epochs):
        print(epoch)

        model.train()
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            src, tgt, src_pad_mask, tgt_pad_mask = batch
            src, tgt, src_pad_mask, tgt_pad_mask = src.to(device).cuda(), tgt.to(device).cuda(), \
                                                   src_pad_mask.to(device).cuda(), tgt_pad_mask.to(device).cuda()

            tgt_input = tgt[:, :-1] # Remove last EOS
            tgt_output = tgt[:, 1:]  # Remove first SOS
            
            outputs = model(src, tgt_input, src_pad_mask, tgt_pad_mask)

            optimizer.zero_grad()
            loss = 0
            for idx, out in enumerate(outputs):
                # Flatten
                tgt_output_tok = tgt_output[:, :, idx].reshape(-1)
                out_reshaped = out.reshape(-1, out.shape[-1])

                if cfg["transformer"].get("bayes_compression"):
                    KLD = model._kl_divergence()
                    loss += objective(criterion, out_reshaped, tgt_output_tok, KLD)
                else:
                    loss += criterion(out_reshaped, tgt_output_tok)

            loss.backward()

            optimizer.step()

            # Tensorboard
            global_step = epoch*len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), global_step)

        # Decode example
        if decode and epoch % decode_every == 0:
            file_path = os.path.join(decode_dir, 'train_'+str(epoch))
            tokens = greedy_decode(file_path, model, train_loader, PAD_IDX, SOS_IDX, EOS_IDX, device, save_src=(epoch==0))
            file_path = os.path.join(decode_dir, 'test_'+str(epoch))
            tokens = greedy_decode(file_path, model, test_loader, PAD_IDX, SOS_IDX, EOS_IDX, device, save_src=(epoch==0))
            if tokens is None:
                writer.add_text('Decoded/train', "BadMidi", global_step)
            else:
                writer.add_text('Decoded/train', str(tokens), global_step)

        # Validation loop
        model.eval()
        val_loss = 0
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            with torch.no_grad():
                src, tgt, src_pad_mask, tgt_pad_mask = batch
                src, tgt, src_pad_mask, tgt_pad_mask = src.to(device).cuda(), tgt.to(device).cuda(), \
                                                    src_pad_mask.to(device).cuda(), tgt_pad_mask.to(device).cuda()

                tgt_input = tgt[:, :-1] # Remove last EOS
                tgt_output = tgt[:, 1:]  # Remove first SOS

                outputs = model(src, tgt_input, src_pad_mask, tgt_pad_mask)

                for idx, out in enumerate(outputs):
                    # Flatten
                    tgt_output_tok = tgt_output[:, :, idx].reshape(-1)
                    out_reshaped = out.reshape(-1, out.shape[-1])

                    if cfg["transformer"].get("bayes_compression"):
                        KLD = model._kl_divergence()
                        val_loss += objective(criterion, out_reshaped, tgt_output_tok, KLD)
                    else:
                        val_loss += criterion(out_reshaped, tgt_output_tok)

        # Tensorboard
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, global_step)

        # Save model
        if save and epoch % save_every == 0:
            save_loc = os.path.join(save_dir, "model_epoch_{:}.pt".format(epoch))
            torch.save(model.state_dict(), save_loc)


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
    train(cfg_file=args.config)
