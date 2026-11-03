# -*- coding: utf-8 -*-

"""
@author: Yang Cui, Uppsala University

Testing script for U-FNO source-to-wavefield prediction.
Input: source location [x, y, z], Output: [T, X, Y, Z]
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from ufno3d import UFNO3d
from train_ufno import encode_source


# Parameters Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
source_path = os.path.join(data_dir, "test_sources.npy")
true_path = os.path.join(data_dir, "test_snapshots.npy")  # None for sources without labels.
ckpt_path = os.path.join(base_dir, "checkpoints", "ufno3d_best.pth")
results_dir = os.path.join(base_dir, "results")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# Evaluation metrics
def relative_l2(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)


def mse(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return np.mean((pred - true) ** 2)


# Inference
@torch.no_grad()
def predict(model, source_loc, config):
    """Predict from source location using the scale saved during training."""
    model.eval()
    source = encode_source(source_loc, config["grid_shape"], config["source_sigma"])
    device = next(model.parameters()).device
    x = torch.tensor(source[None], dtype=torch.float32, device=device)

    pred = model(x)
    pred = pred.squeeze(0).cpu().numpy()
    pred = pred * config["wavefield_scale"]
    return pred


# Results visualization
def visualise_comparison(pred, true, src_i, source_loc, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    snap_ids = sorted(set([pred.shape[0] // 2, pred.shape[0] - 1]))
    center = pred.shape[-1] // 2
    n_cols = 1 if true is None else 3

    fig, axes = plt.subplots(len(snap_ids), n_cols, figsize=(5 * n_cols, 4 * len(snap_ids)),
                             squeeze=False)
    for row, sidx in enumerate(snap_ids):
        if true is None:
            reference = pred[sidx]
            panels = [("Prediction", pred[sidx])]
        else:
            reference = true[sidx]
            panels = [("Ground Truth", true[sidx]),
                      ("Prediction", pred[sidx]),
                      ("Error (true - pred)", true[sidx] - pred[sidx])]

        vmax = max(float(np.percentile(np.abs(reference), 99)), 1e-12)
        for ax, (title, cube) in zip(axes[row], panels):
            im = ax.imshow(cube[:, :, center].T, cmap="seismic", vmin=-vmax,
                           vmax=vmax, origin="lower", aspect="equal")
            ax.set_title(f"{title} | snap={sidx} | z={center}")
            ax.set_xlabel("X grid index")
            ax.set_ylabel("Y grid index")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Source {src_i} | grid location={tuple(source_loc)}")
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"ufno_compare_src{src_i}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# Main
def main():
    os.makedirs(results_dir, exist_ok=True)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"\n[test] DEVICE: {device}")

    # Load source locations
    test_sources = np.load(source_path)
    print(f"test_sources shape = {test_sources.shape}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = UFNO3d(
        modes1=cfg["modes"], modes2=cfg["modes"], modes3=cfg["modes"],
        width=cfg["width"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint (epoch={ckpt['epoch']})")

    # Load labels only for evaluation
    true_all = None
    if true_path is not None:
        true_all = np.load(true_path, mmap_mode='r')

    # Prediction
    n_test = len(test_sources)
    output_shape = (n_test, cfg["out_channels"], *cfg["grid_shape"])
    output_path = os.path.join(results_dir, "ufno_pred_snapshots.npy")
    pred_all = np.lib.format.open_memmap(output_path, mode='w+', dtype=np.float32,
                                         shape=output_shape)
    metrics = []
    print("\n[test] Running prediction ...")
    for i, loc in enumerate(test_sources):
        pred = predict(model, loc, cfg)
        pred_all[i] = pred

        true = None
        if true_all is not None:
            true = true_all[i]
            rl2 = relative_l2(pred, true)
            ms = mse(pred, true)
            metrics.append((i, rl2, ms))
            print(f"Source {i:2d} | loc={loc} | Rel-L2={rl2:.6f} | MSE={ms:.6e}")
        else:
            print(f"Source {i:2d} | loc={loc} | predicted")

        visualise_comparison(pred, true, i, loc, fig_dir)

    # Save results
    pred_all.flush()
    np.save(os.path.join(results_dir, "source_locations.npy"), test_sources)
    if true_all is not None:
        np.savetxt(os.path.join(results_dir, "metrics.csv"), metrics, delimiter=",",
                   header="source,relative_l2,mse", comments="", fmt=["%d", "%.8e", "%.8e"])

    print(f"\n[test] Saved: {output_path}")
    print("\n[test] Done.")


if __name__ == '__main__':
    main()
