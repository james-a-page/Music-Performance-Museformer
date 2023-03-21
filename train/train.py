import os
import argparse
import shutil

from dataset import load_data
from utils import set_seed, load_config
from models.model.transformer import Transformer
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter


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

    train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX = load_data(data_cfg=cfg["data"])

    model = Transformer(cfg["transformer"], len(tokenizer.vocab._token_to_event), SOS_IDX, PAD_IDX, device).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["transformer"].get("lr")))
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

        model.train()
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            with torch.autocast(device, dtype=torch.float16):
                src, tgt = batch
                src, tgt = src.to(device), tgt.to(device)

                src = src[:-1, :] # Remove last EOS
                tgt = tgt[1:, :]  # Remove first SOS

                logits = model(src, None)

                optimizer.zero_grad()
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
                loss.backward()

                optimizer.step()

                # Tensorboard
                global_step = epoch*len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss.item(), global_step)


        model.eval()
        val_loss = 0
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            with torch.no_grad():
                with torch.autocast(device, dtype=torch.float16):
                    src, tgt = batch
                    src, tgt = src.to(device), tgt.to(device)

                    logits = model(src, None)

                    optimizer.zero_grad()
                    val_loss += criterion(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)).item()

        # Tensorboard
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, global_step)


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
