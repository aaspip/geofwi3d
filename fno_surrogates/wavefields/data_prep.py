# -*- coding: utf-8 -*-
"""
@author: Yang Cui, Uppsala University

Generate 3D wavefield snapshots using Deepwave for FNO training and testing.
Output: train_snapshots.npy, test_snapshots.npy, train_sources.npy, test_sources.npy
"""

import os
import numpy as np
import torch
import deepwave
from deepwave import scalar

# Parameters Setup
data_root = '../allmodels/'
model_folders = [f'model_{i:04d}' for i in range(100, 200)]
model_index = 2

data_out_dir = './data'

nx, ny, nz = 96, 96, 96
dx = 10.0

dt = 0.001
nt = 1000

freq = 5.0

accuracy = 4
pml_width = 20

n_snaps = 20
snap_interval = nt // n_snaps

n_train = 30
n_test = 5
safe_margin = pml_width/2 + 5

seed = 202506
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[data_prep] Using device: {device}")


# Utils
def read_models(data_root, model_folders, index):
    vp_file = os.path.join(data_root, model_folders[index], "vp3d.bin")
    shape = (nx, ny, nz)
    vp = np.memmap(vp_file, dtype=np.float32, mode='r', shape=shape)
    return np.array(vp)


def ricker_wavelet(freq, dt, nt):
    t = np.arange(nt) * dt
    t0 = 1.0 / freq
    u = np.pi * freq * (t - t0)
    w = (1.0 - 2.0 * u**2) * np.exp(-u**2)
    return w.astype(np.float32)


def random_source_locations(n_sources, nx, ny, nz, margin, seed=None):
    rng = np.random.default_rng(seed)
    lo, hi = margin, [nx - margin, ny - margin, nz - margin]
    locs = rng.integers(low=lo, high=hi, size=(n_sources, 3), dtype=np.int32)
    return locs


def simulate_one_source(vp_tensor, src_loc, wavelet_tensor):
    ix, iy, iz = int(src_loc[0]), int(src_loc[1]), int(src_loc[2])
    
    src_loc_t = torch.tensor([[[ix, iy, iz]]], dtype=torch.long, device=device)
    src_amp = wavelet_tensor.unsqueeze(0).unsqueeze(0)
    
    snapshots = []
    
    def _snap_callback(state):
        wf = state.get_wavefield("wavefield_0")
        snapshots.append(wf[0].detach().cpu().numpy().astype(np.float32))
    
    deepwave.scalar(
        v=vp_tensor,
        grid_spacing=dx,
        dt=dt,
        source_amplitudes=src_amp,
        source_locations=src_loc_t,
        accuracy=accuracy,
        pml_width=pml_width,
        forward_callback=_snap_callback,
        callback_frequency=snap_interval,
    )
    
    if len(snapshots) < n_snaps:
        print(f"  WARNING: got {len(snapshots)} snapshots, expected {n_snaps}")
    snapshots = snapshots[:n_snaps]
    
    return np.stack(snapshots, axis=0)


def run_simulations(vp_tensor, source_locs, wavelet_tensor, tag="train"):
    n = len(source_locs)
    all_snaps = np.zeros((n, n_snaps, nx, ny, nz), dtype=np.float32)
    
    for i, loc in enumerate(source_locs):
        print(f"  [{tag}] source {i+1}/{n}  loc=({loc[0]},{loc[1]},{loc[2]})")
        snaps = simulate_one_source(vp_tensor, loc, wavelet_tensor)
        all_snaps[i] = snaps
    
    return all_snaps


# Main Code
def main():
    os.makedirs(data_out_dir, exist_ok=True)
    
    # Load velocity model
    print("[data_prep] Reading velocity model …")
    vp_np = read_models(data_root, model_folders, model_index)
    print(f"  Velocity range: {vp_np.min():.1f} – {vp_np.max():.1f} m/s")
    
    # CFL check
    v_max = float(vp_np.max())
    dt_max = dx / (np.sqrt(3) * v_max)
    assert dt <= dt_max, f"CFL violated: dt={dt:.5f} > dt_max={dt_max:.5f}"
    print(f"  CFL OK (dt_max={dt_max:.5f} s)")
    
    vp_tensor = torch.tensor(vp_np, dtype=torch.float32, device=device)
    
    # Build Ricker wavelet
    wavelet = ricker_wavelet(freq, dt, nt)
    wav_t = torch.tensor(wavelet, dtype=torch.float32, device=device)
    
    # Generate random source locations
    all_locs = random_source_locations(n_train + n_test, nx, ny, nz, safe_margin, seed=seed)
    train_locs = all_locs[:n_train]
    test_locs = all_locs[n_train:]
    
    np.save(os.path.join(data_out_dir, 'train_sources.npy'), train_locs)
    np.save(os.path.join(data_out_dir, 'test_sources.npy'), test_locs)
    print(f"  Saved source locations (train={n_train}, test={n_test})")
    
    # Training simulations
    print("[data_prep] Running TRAINING simulations …")
    train_snaps = run_simulations(vp_tensor, train_locs, wav_t, tag="train")
    np.save(os.path.join(data_out_dir, 'train_snapshots.npy'), train_snaps)
    print(f"  Saved train_snapshots.npy shape={train_snaps.shape}")
    
    # Testing simulations
    print("[data_prep] Running TESTING simulations …")
    test_snaps = run_simulations(vp_tensor, test_locs, wav_t, tag="test")
    np.save(os.path.join(data_out_dir, 'test_snapshots.npy'), test_snaps)
    print(f"  Saved test_snapshots.npy shape={test_snaps.shape}")
    
    # Summary
    print(f"\n[data_prep] Done. train_snapshots: {train_snaps.shape} ({train_snaps.nbytes/1e9:.2f} GB)")
    print(f"  test_snapshots: {test_snaps.shape} ({test_snaps.nbytes/1e9:.2f} GB)")


if __name__ == '__main__':
    main()