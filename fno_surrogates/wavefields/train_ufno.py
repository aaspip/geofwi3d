# -*- coding: utf-8 -*-

"""
@author: Yang Cui, Uppsala University

Training script for U-FNO source-to-wavefield prediction.
Input: [B, 1, X, Y, Z], Output: [B, T, X, Y, Z]
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ufno3d import UFNO3d


# Parameters Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
ckpt_dir = os.path.join(base_dir, "checkpoints")
log_dir = os.path.join(base_dir, "logs")

in_channels = 1
width = 36
modes = 8
source_sigma = 1.5

batch_size = 1
epochs = 200
lr = 5e-4
weight_decay = 1e-4
grad_accum = 4
seed = 202605

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# Source location encoding
def encode_source(source_loc, grid_shape, sigma):
    """Encode source grid indices [x, y, z] as a Gaussian volume [1, X, Y, Z]."""
    source_loc = np.asarray(source_loc, dtype=np.float32)
    x = np.arange(grid_shape[0], dtype=np.float32)[:, None, None]
    y = np.arange(grid_shape[1], dtype=np.float32)[None, :, None]
    z = np.arange(grid_shape[2], dtype=np.float32)[None, None, :]

    distance_sq = ((x - source_loc[0]) ** 2 +
                   (y - source_loc[1]) ** 2 +
                   (z - source_loc[2]) ** 2)
    source = np.exp(-distance_sq / (2 * sigma ** 2))
    return source[None].astype(np.float32)


# Wavefield normalization
def wavefield_scale(snapshots, indices):
    """Compute one RMS scale from training sources only."""
    total_sq = 0.0
    count = 0

    for idx in indices:
        wavefield = np.asarray(snapshots[idx], dtype=np.float64)
        total_sq += np.sum(wavefield ** 2)
        count += wavefield.size

    return float(np.sqrt(total_sq / count))


class WavefieldDataset(Dataset):
    def __init__(self, snapshots, sources, indices, scale, sigma):
        """snapshots: [Nsrc, T, X, Y, Z], sources: [Nsrc, 3]"""
        super().__init__()
        self.snapshots = snapshots
        self.sources = sources
        self.indices = np.asarray(indices)
        self.scale = scale
        self.sigma = sigma
        self.grid_shape = tuple(snapshots.shape[2:])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = self.indices[idx]
        x = encode_source(self.sources[source_idx], self.grid_shape, self.sigma)
        y = np.array(self.snapshots[source_idx], dtype=np.float32, copy=True)
        y = y / self.scale
        return (torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))


# Training loop
def train_one_epoch(model, loader, optimizer, criterion, accum_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        # Use the actual sample count, including the last accumulation group.
        group_start = (step // accum_steps) * accum_steps
        group_samples = min(accum_steps * loader.batch_size,
                            len(loader.dataset) - group_start * loader.batch_size)
        scaled_loss = loss * x.shape[0] / group_samples
        scaled_loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * x.shape[0]

    return total_loss / len(loader.dataset)


# Validation loop
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.shape[0]

    return total_loss / len(loader.dataset)


# Main
def main():
    torch.manual_seed(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n[train] DEVICE: {device}")

    # Load data
    print("\n[train] Loading data ...")
    train_snapshots = np.load(os.path.join(data_dir, "train_snapshots.npy"), mmap_mode='r')
    train_sources = np.load(os.path.join(data_dir, "train_sources.npy"))
    print(f"train_snapshots shape = {train_snapshots.shape}")
    print(f"train_sources shape = {train_sources.shape}")

    # Train/val split by source
    n_total = train_snapshots.shape[0]
    n_val = max(1, int(0.2 * n_total))
    indices = np.random.default_rng(seed).permutation(n_total)
    train_indices = indices[:-n_val]
    val_indices = indices[-n_val:]

    # One shared scale; validation and test targets do not set the scale.
    scale = wavefield_scale(train_snapshots, train_indices)
    grid_shape = tuple(train_snapshots.shape[2:])
    out_channels = train_snapshots.shape[1]

    ds_train = WavefieldDataset(train_snapshots, train_sources, train_indices, scale, source_sigma)
    ds_val = WavefieldDataset(train_snapshots, train_sources, val_indices, scale, source_sigma)
    print(f"\nTrain samples = {len(ds_train)}")
    print(f"Val samples = {len(ds_val)}")
    print(f"Wavefield RMS scale = {scale:.6e}")

    # Dataloader
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0,
                          pin_memory=device.type == 'cuda')
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda')

    # Model
    model = UFNO3d(
        modes1=modes, modes2=modes, modes3=modes,
        width=width,
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)
    print(f"\nUFNO parameters: {model.count_params()/1e6:.2f} M")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.L1Loss()

    # Logging
    log_path = os.path.join(log_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,lr,time_s\n")

    best_val = np.inf
    ckpt_path = os.path.join(ckpt_dir, "ufno3d_best.pth")
    t0_global = time.time()

    # Training loop
    print("\n[train] Starting training ...\n")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, dl_train, optimizer, criterion, grad_accum)
        val_loss = validate(model, dl_val, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:4d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e} | {elapsed:.1f}s")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{lr_now:.6e},{elapsed:.2f}\n")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "modes": modes,
                    "width": width,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "grid_shape": grid_shape,
                    "source_sigma": source_sigma,
                    "wavefield_scale": scale,
                },
                "train_indices": train_indices.tolist(),
                "val_indices": val_indices.tolist(),
                "seed": seed,
            }, ckpt_path)
            print(f" [BEST] Saved model (val={best_val:.6f})")

    # Finish
    total_time = time.time() - t0_global
    print("\nTraining complete")
    print(f"Best validation loss = {best_val:.6f}")
    print(f"Total time = {total_time/60:.1f} min")
    print(f"Checkpoint = {ckpt_path}")
    print(f"Log file = {log_path}")


if __name__ == '__main__':
    main()
