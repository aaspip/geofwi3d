# -*- coding: utf-8 -*-
"""
@author: Yang Cui, Uppsala University

Testing script for modified U-FNO 3D seismic wavefield model.
Input: [Nsrc, T, 96, 96, 96], Output: [Nsrc, T, 96, 96, 96]
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fno3d'))
from ufno3d import UFNO3d


# Parameters Setup
data_dir = "./data"
ckpt_path = "./checkpoints/ufno3d_best.pth"
results_dir = "./results"
fig_dir = os.path.join(results_dir, "figures")

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[test] device: {device}")


# wavefield normalization
def normalize_wavefield(x):
    mean = x.mean()
    std = x.std()
    if std < 1e-8:
        return x - mean, mean, std
    return (x - mean) / std, mean, std


def denormalize_wavefield(x, mean, std):
    if std < 1e-8:
        return x + mean
    return x * std + mean


# evaluation metrics
def relative_l2(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)


def mse(pred, true):
    return np.mean((pred - true) ** 2)


def ssim_3d_slice(pred, true, frame_idx):
    try:
        from skimage.metrics import structural_similarity as ssim
        nz = pred.shape[-1]
        p = pred[frame_idx, :, :, nz // 2]
        t = true[frame_idx, :, :, nz // 2]
        drange = max(t.max() - t.min(), 1e-12)
        return ssim(p, t, data_range=drange)
    except:
        return np.nan


# inference
@torch.no_grad()
def predict(model, wavefield):
    """wavefield: [T, X, Y, Z], returns: [T, X, Y, Z]"""
    model.eval()
    
    wavefield_norm, mean, std = normalize_wavefield(wavefield)
    x = torch.tensor(wavefield_norm[None], dtype=torch.float32, device=device)
    
    pred = model(x)
    pred = pred.squeeze(0).cpu().numpy()
    pred = denormalize_wavefield(pred, mean, std)
    
    return pred


# results visualization 
def visualise_comparison(pred_all, true_all, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    n_src = pred_all.shape[0]
    n_snap = pred_all.shape[1]
    snap_ids = [n_snap // 2, n_snap - 5]
    nz = pred_all.shape[-1]
    center = nz // 2
    
    for src_i in range(n_src):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for row in range(2):
            sidx = snap_ids[row]
            true_frame = true_all[src_i, sidx]
            pred_frame = pred_all[src_i, sidx]
            vmax = np.percentile(np.abs(true_frame), 99)
            vmin = -vmax
            
            axes[row, 0].imshow(true_frame[:, :, center].T, cmap="seismic", vmin=vmin, vmax=vmax, origin="lower")
            axes[row, 0].set_title(f"GT | src={src_i} | snap={sidx}")
            
            axes[row, 1].imshow(pred_frame[:, :, center].T, cmap="seismic", vmin=vmin, vmax=vmax, origin="lower")
            axes[row, 1].set_title(f"UFNOPred | src={src_i} | snap={sidx}")
        
        fig.tight_layout()
        save_path = os.path.join(save_dir, f"ufnocompare_src{src_i}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")


def visualise_comparison_3d(pred_all, true_all, save_dir):
    try:
        from plotting import plot3d
    except ImportError:
        print("\n[vis] plotting.plot3d not found")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    n_src = pred_all.shape[0]
    n_snap = pred_all.shape[1]
    snap_ids = [n_snap // 2, n_snap - 1]
    center = pred_all.shape[-1] // 2
    
    for src_i in range(n_src):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={"projection": "3d"}, constrained_layout=True)
        
        titles = [
            ("Ground Truth", true_all, snap_ids[0]),
            ("Prediction", pred_all, snap_ids[0]),
            ("Ground Truth", true_all, snap_ids[1]),
            ("Prediction", pred_all, snap_ids[1]),
        ]
        
        for ax, (title, arr, sidx) in zip(axes.ravel(), titles):
            plt.sca(ax)
            cube = arr[src_i, sidx]
            cube_t = cube.transpose(2, 0, 1)  # [X,Y,Z] -> [Z,X,Y]
            vmax = np.percentile(np.abs(true_all[src_i, sidx]), 99)
            
            plot3d(cube_t, cmap="seismic", vmin=-vmax, vmax=vmax,
                   dx=10.0, dy=10.0, dz=10.0,
                   frames=[center, center, center],
                   ifinside=False, ifslice=True,
                   showf=False, close=False, ifnewfig=False, nlevel=100)
            
            ax.set_title(f"{title} | src={src_i} | snap={sidx}", fontsize=10)
        
        save_path = os.path.join(save_dir, f"ufno_compare_src{src_i}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


# main
def main():
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load test data
    print("\n[test] Loading test data ...")
    test_snapshots = np.load(os.path.join(data_dir, "test_snapshots.npy"))
    print(f"test_snapshots shape = {test_snapshots.shape}")
    
    n_snapshots = test_snapshots.shape[1]
    
    # Load checkpoint
    print(f"\n[test] Loading model ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    
    model = UFNO3d(
        modes1=cfg["modes"], modes2=cfg["modes"], modes3=cfg["modes"],
        width=cfg["width"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"]
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint (epoch={ckpt['epoch']})")
    
    # Prediction
    n_test = test_snapshots.shape[0]
    pred_all = np.zeros_like(test_snapshots)
    
    print("\n[test] Running prediction ...")
    for i in range(n_test):
        pred_all[i] = predict(model, test_snapshots[i])
        rl2 = relative_l2(pred_all[i], test_snapshots[i])
        ms = mse(pred_all[i], test_snapshots[i])
        print(f"Source {i:2d} | Rel-L2 = {rl2:.6f} | MSE = {ms:.6e}")
    
    # Save results
    np.save(os.path.join(results_dir, "ufno_pred_snapshots.npy"), pred_all)
    np.save(os.path.join(results_dir, "ufno_true_snapshots.npy"), test_snapshots)
    print("\n[test] Saved prediction results")
    
    # Metrics
    print("\n[test] Evaluation Summary")
    print(f"\n{'Src':>4} {'Rel-L2':>12} {'MSE':>12} {'SSIM':>12}")
    
    for i in range(n_test):
        rl2 = relative_l2(pred_all[i], test_snapshots[i])
        ms = mse(pred_all[i], test_snapshots[i])
        ss = ssim_3d_slice(pred_all[i], test_snapshots[i], n_snapshots // 2)
        print(f"{i:4d} {rl2:12.6f} {ms:12.4e} {ss:12.6f}")
    
    # Visualization
    print("\n[test] Generating figures ...")
    visualise_comparison_3d(pred_all, test_snapshots, fig_dir)
    print("\n[test] Done.")


if __name__ == "__main__":
    main()