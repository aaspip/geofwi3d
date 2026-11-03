# -*- coding: utf-8 -*-

"""
@author: Yang Cui, Uppsala University

Generate training and testing wavefields in one fixed velocity model.
Output: source locations [Nsrc, 3] and wavefields [Nsrc, T, X, Y, Z].
"""

import os

import numpy as np
import torch
import deepwave


# Parameters Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
velocity_path = os.path.join(base_dir, "data", "velocity_model.npy")
data_out_dir = os.path.join(base_dir, "data")

nx = ny = nz = 96
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
safe_margin = pml_width // 2 + 5
seed = 202506
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# Utils
def ricker_wavelet(freq, dt, nt):
    t = np.arange(nt) * dt
    t0 = 1.0 / freq
    u = np.pi * freq * (t - t0)
    w = (1.0 - 2.0 * u ** 2) * np.exp(-u ** 2)
    return w.astype(np.float32)


def random_source_locations(n_sources, nx, ny, nz, margin, seed=None):
    rng = np.random.default_rng(seed)
    lo = margin
    hi = [nx - margin, ny - margin, nz - margin]
    return rng.integers(low=lo, high=hi, size=(n_sources, 3), dtype=np.int32)


@torch.no_grad()
def simulate_one_source(vp_tensor, src_loc, wavelet_tensor):
    ix, iy, iz = int(src_loc[0]), int(src_loc[1]), int(src_loc[2])
    src_loc_t = torch.tensor([[[ix, iy, iz]]], dtype=torch.long, device=device)
    src_amp = wavelet_tensor.unsqueeze(0).unsqueeze(0)
    snapshots = []
    snapshot_times = []

    def _snap_callback(state):
        wf = state.get_wavefield("wavefield_0", view="inner")
        snapshots.append(wf[0].detach().cpu().numpy().astype(np.float32))
        snapshot_times.append(state.step * dt)

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
    return np.stack(snapshots[:n_snaps]), np.array(snapshot_times[:n_snaps])


def run_simulations(vp_tensor, source_locs, wavelet_tensor, tag="train"):
    n = len(source_locs)
    output_path = os.path.join(data_out_dir, f"{tag}_snapshots.npy")
    all_snaps = np.lib.format.open_memmap(output_path, mode='w+', dtype=np.float32,
                                          shape=(n, n_snaps, nx, ny, nz))
    for i, loc in enumerate(source_locs):
        print(f"  [{tag}] source {i+1}/{n}  loc=({loc[0]},{loc[1]},{loc[2]})")
        snaps, snapshot_times = simulate_one_source(vp_tensor, loc, wavelet_tensor)
        all_snaps[i] = snaps

    all_snaps.flush()
    print(f"  Saved {tag}_snapshots.npy shape={all_snaps.shape}")
    return snapshot_times


# Main
def main():
    os.makedirs(data_out_dir, exist_ok=True)
    print(f"Using device: {device}")

    # Load velocity model
    vp_np = np.load(velocity_path)
    vp_tensor = torch.tensor(vp_np, dtype=torch.float32, device=device)
    print(f"  Velocity range: {vp_np.min():.1f} - {vp_np.max():.1f} m/s")

    # Source time function
    wavelet = ricker_wavelet(freq, dt, nt)
    wav_t = torch.tensor(wavelet, dtype=torch.float32, device=device)

    # Generate source locations
    all_locs = random_source_locations(n_train + n_test, nx, ny, nz, safe_margin, seed=seed)
    train_locs = all_locs[:n_train]
    test_locs = all_locs[n_train:]
    np.save(os.path.join(data_out_dir, "train_sources.npy"), train_locs)
    np.save(os.path.join(data_out_dir, "test_sources.npy"), test_locs)

    # Training simulations
    print("Running Training data simulations ...")
    snapshot_times = run_simulations(vp_tensor, train_locs, wav_t, tag="train")

    # Testing simulations
    print("Running Testing data simulations ...")
    run_simulations(vp_tensor, test_locs, wav_t, tag="test")
    np.save(os.path.join(data_out_dir, "snapshot_times.npy"), snapshot_times)
    print("\n[data_prep] Done.")


if __name__ == '__main__':
    main()
