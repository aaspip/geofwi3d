# -*- coding: utf-8 -*-
"""
@author: Yang Cui, Uppsala University

Training script for modified U-FNO 3D seismic wavefield model.
Input: [Nsrc, T, 96, 96, 96], Output: [Nsrc, T, 96, 96, 96]
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fno3d'))
from ufno3d import UFNO3d


# Parameters Setup
data_dir = "./data"
ckpt_dir = "./checkpoints"
log_dir = "./logs"

nx = ny = nz = 96
num_snapshots = 20

in_channels = num_snapshots
out_channels = num_snapshots
width = 36
modes = 8

batch_size = 1
epochs = 200
learning_rate = 5e-4
weight_decay = 1e-4
grad_accum = 4
seed = 202605

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[train] device: {device}")


# wavefield normalization
def normalize_wavefield(x):
    """Normalize wavefield: (x - mean) / std"""
    mean = x.mean()
    std = x.std()
    if std < 1e-8:
        return x - mean
    return (x - mean) / std

class WavefieldDataset(Dataset):
    def __init__(self, snapshots):
        """snapshots: [Nsrc, T, 96, 96, 96]"""
        super().__init__()
        self.snapshots = snapshots

    def __len__(self):
        return self.snapshots.shape[0]

    def __getitem__(self, idx):
        wavefield = normalize_wavefield(self.snapshots[idx])
        X = wavefield.copy()
        Y = wavefield.copy()
        return (torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))


# Training loop
def train_one_epoch(model, loader, optimizer, criterion, accum_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for step, (X, Y) in enumerate(loader):
        X = X.to(device)
        Y = Y.to(device)
        
        pred = model(X)
        loss = criterion(pred, Y) / accum_steps
        loss.backward()
        
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
    
    return total_loss / len(loader)


# validation loop
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        pred = model(X)
        loss = criterion(pred, Y)
        total_loss += loss.item()
    return total_loss / len(loader)


# main 
def main():
    torch.manual_seed(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    print("\n[train] Loading data ...")
    train_snapshots = np.load(os.path.join(data_dir, "train_snapshots.npy"))
    print(f"\ntrain_snapshots shape = {train_snapshots.shape}")
    assert train_snapshots.ndim == 5
    assert train_snapshots.shape[2] == 96
    
    # Train/val split
    n_total = train_snapshots.shape[0]
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    
    train_data = train_snapshots[:n_train]
    val_data = train_snapshots[n_train:]
    
    ds_train = WavefieldDataset(train_data)
    ds_val = WavefieldDataset(val_data)
    
    print(f"\nTrain samples = {len(ds_train)}")
    print(f"Val samples   = {len(ds_val)}")
    
    # Dataloader
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = UFNO3d(
        modes1=modes, modes2=modes, modes3=modes,
        width=width,
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)
    
    print(f"\nUFNO parameters: {model.count_params()/1e6:.2f} M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
    criterion = nn.L1Loss()
    
    # Logging
    log_path = os.path.join(log_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,lr,time_s\n")
    
    best_val = np.inf
    t0_global = time.time()
    
    # Training loop
    print(f"\n[train] Starting training ...\n")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        train_loss = train_one_epoch(model, dl_train, optimizer, criterion, grad_accum)
        val_loss = validate(model, dl_val, criterion)
        scheduler.step()
        
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch:4d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e} | {elapsed:.1f}s")
        
        # Save log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{lr_now:.6e},{elapsed:.2f}\n")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(ckpt_dir, "ufno3d_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "modes": modes,
                    "width": width,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                }
            }, ckpt_path)
            print(f" [BEST] Saved model (val={best_val:.6f})")
    
    # Finish
    total_time = time.time() - t0_global
    print(f"\nTraining complete")
    print(f"Best validation loss = {best_val:.6f}")
    print(f"Total time = {total_time/60:.1f} min")
    print(f"Checkpoint = {ckpt_path}")
    print(f"Log file   = {log_path}")


if __name__ == "__main__":
    main()