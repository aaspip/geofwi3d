#!/usr/bin/env python3
"""
FWI optimizer comparison (Deepwave), converted from test_optimizers.ipynb.

Runs multiscale (10 / 15 / 25 Hz band-pass) L2 waveform inversion with several
optimizers in parallel: Adam, AdamW, RMSprop, and SGD.
(L-BFGS code is commented out below — re-enable OPT_NAMES / make_lbfgs_closure to use it.)

Fixes applied vs original:
  1. Loss / backward use weight batch_size / n_shots so uneven last minibatches match
     the full objective.
  2. pml_width[0] = 0: free surface on the top boundary; other faces use PML (20 cells).
  3. First-order optimizers use --grad-storage device (fast).
  4. gaussian_filter uses default SciPy truncation (no radius= override).
  5. Observed traces: obs_filt_cpu is filtered *once* per band; minibatch slices are
     loaded to GPU without re-running apply_filt (avoids double band-pass).

Results: velocities and per-optimizer loss curves are saved to a single .npz (--output).
A 2×3 comparison figure (true, init, Adam / AdamW / RMSprop / SGD) is written to
--figure (default: optimizers-fwi.pdf) unless --no-plot is set.

Plot from .npz:
  python test_optimizers.py plot [fwi_optimizers.npz] [optimizers-fwi.pdf]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
from torchaudio.functional import biquad

import deepwave
from deepwave import scalar

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "quick_start"))
from plotting import plot3d  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--velocity",
        type=Path,
        default=Path("../allmodels/model_1693/vp3d.bin"),
        help="float32 (nx,ny,nz) binary velocity path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("fwi_optimizers.npz"),
        help="Where to save inversion results (.npz)",
    )
    p.add_argument(
        "--sigma-init",
        type=float,
        default=10.0,
        help="Gaussian smooth sigma for v_init",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: auto)",
    )
    p.add_argument(
        "--grad-storage",
        choices=("cpu", "device"),
        default="device",
        help="Deepwave storage for Adam / AdamW / RMSprop / SGD scalar() (requires_grad). "
        '"device" is much faster; use "cpu" only if those steps OOM.',
    )
    p.add_argument(
        "--shots-per-batch",
        type=int,
        default=2,
        help="Shots per minibatch (smaller uses less peak GPU memory).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the 3D velocity comparison figure after saving results.",
    )
    p.add_argument(
        "--figure",
        type=Path,
        default=Path("optimizers-fwi.pdf"),
        help="Output path for the optimizer comparison figure (PDF/PNG).",
    )
    p.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the figure interactively (in addition to saving).",
    )
    return p.parse_args()


def taper(x: torch.Tensor) -> torch.Tensor:
    return deepwave.common.cosine_taper_end(x, 50)


def get_filter(cutoff: float, dt: float, device: torch.device):
    sos = butter(4, cutoff, fs=1 / dt, output="sos")
    return [torch.from_numpy(s).to(device).to(torch.float32) for s in sos]


def apply_filt(x: torch.Tensor, sos_list: list) -> torch.Tensor:
    for s in sos_list:
        x = biquad(x, s[0], s[1], s[2], s[3], s[4], s[5])
    return x


class Model(nn.Module):
    """Logit parameterization of velocity in [min_vel, max_vel]."""

    def __init__(self, initial: torch.Tensor, min_vel: float, max_vel: float):
        super().__init__()
        self.min_vel, self.max_vel = min_vel, max_vel
        normalized = (initial - min_vel) / (max_vel - min_vel)
        safe = torch.clamp(normalized, min=1e-4, max=1.0 - 1e-4)
        self.model = nn.Parameter(torch.logit(safe))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.model) * (self.max_vel - self.min_vel) + self.min_vel


# AdamW: large enough to matter vs loss_multiplier-scaled waveform gradients on logits.
ADAMW_WEIGHT_DECAY = 0.1


def build_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=ADAMW_WEIGHT_DECAY)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=0.0)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=False)
    raise ValueError(f"Unknown optimizer: {name}")


FIRST_ORDER = ("adam", "adamw", "rmsprop", "sgd")
OPT_NAMES = FIRST_ORDER 




def plot_optimizer_comparison(
    vp_true: np.ndarray,
    vp_init: np.ndarray,
    vp_adam: np.ndarray,
    vp_adamw: np.ndarray,
    vp_rmsprop: np.ndarray,
    vp_sgd: np.ndarray,
    figure_path: Path,
    *,
    show: bool = False,
) -> None:
    """2×3 panel comparison of true, initial, and per-optimizer inverted velocities."""
    frames = [65, 45, 45]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    plot_items = [
        (vp_true, "True Velocity Model", "a"),
        (vp_init, "Initial Velocity Model", "b"),
        (vp_adam, "Inverted (Adam)", "c"),
        (vp_adamw, "Inverted (AdamW)", "d"),
        (vp_rmsprop, "Inverted (RMSprop)", "e"),
        (vp_sgd, "Inverted (SGD)", "f"),
    ]

    for ax, (vol, title, letter) in zip(axes.ravel(), plot_items):
        ax.set_title(title, pad=10)
        plt.sca(ax)
        plot3d(
            vol,
            cmap="jet",
            vmin=1500,
            vmax=4000,
            dx=10.0,
            dy=10.0,
            dz=10.0,
            ifnewfig=False,
            ifinside=False,
            showf=False,
            close=False,
            frames=frames,
        )
        ax.tick_params(labelsize=10, pad=5)
        ax.set_xlabel("X [m]", labelpad=10)
        ax.set_ylabel("Y [m]", labelpad=10)
        ax.set_zlabel("Z [m]", labelpad=10)
        ax.text2D(
            0.05,
            0.95,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=16,
            va="top",
            ha="left",
            zorder=10,
        )
        ax.set_rasterized(True)

    norm = mcolors.Normalize(vmin=1500, vmax=4000)
    sm = cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(), orientation="vertical", shrink=0.6, pad=0.05
    )
    cbar.set_label("Vp [m/s]", fontsize=14)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure {figure_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_from_npz(npz_path: Path, figure_path: Path, *, show: bool = False) -> None:
    """Load fwi_optimizers.npz and write the comparison figure."""
    vels = np.load(npz_path)
    plot_optimizer_comparison(
        vels["v_true"],
        vels["v_init"],
        vels["v_final_adam"],
        vels["v_final_adamw"],
        vels["v_final_rmsprop"],
        vels["v_final_sgd"],
        figure_path,
        show=show,
    )


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    nz = ny = nx = 96
    vp_path = args.velocity
    if not vp_path.is_file():
        raise FileNotFoundError(f"Velocity file not found: {vp_path.resolve()}")

    v_true = np.memmap(vp_path, dtype=np.float32, mode="r", shape=(nx, ny, nz)).T.copy()
    v_true_tensor = torch.from_numpy(np.ascontiguousarray(v_true)).to(device)

    dz = dy = dx = 10.0
    nt, dt = 501, 0.002
    freq = 25.0
    peak_time = 1.5 / freq

    v_init = gaussian_filter(v_true, sigma=args.sigma_init)
    v_init_tensor = torch.from_numpy(v_init).to(device)

    s_coords_y = torch.linspace(5, ny - 6, 6).long()
    s_coords_x = torch.linspace(5, nx - 6, 6).long()
    n_shots = len(s_coords_y) * len(s_coords_x)
    sy, sx = torch.meshgrid(s_coords_y, s_coords_x, indexing="ij")

    source_locations = torch.zeros(n_shots, 1, 3, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = 1
    source_locations[:, 0, 1] = sy.flatten()
    source_locations[:, 0, 2] = sx.flatten()

    receiver_locations = torch.zeros(n_shots, ny * nx, 3, dtype=torch.long, device=device)
    ry, rx = torch.meshgrid(torch.arange(ny).long(), torch.arange(nx).long(), indexing="ij")
    receiver_locations[..., 0] = 1
    receiver_locations[:, :, 1] = ry.flatten().to(device)
    receiver_locations[:, :, 2] = rx.flatten().to(device)

    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time).repeat(n_shots, 1, 1).to(device)
    )

    # Free surface on top (z-min face): first PML width = 0
    pml_width = [0, 20, 20, 20, 20, 20]

    with torch.no_grad():
        observed_data = scalar(
            v_true_tensor,
            (dz, dy, dx),
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
            pml_width=pml_width,
            storage_mode="device",
        )[-1]
    observed_data = taper(observed_data)
    observed_data_cpu = observed_data.cpu().numpy()
    del observed_data
    torch.cuda.empty_cache()

    n_shots_per_batch = max(1, args.shots_per_batch)
    grad_storage = args.grad_storage
    n_batch = (n_shots + n_shots_per_batch - 1) // n_shots_per_batch
    learning_rate = 1e-2
    mse = nn.MSELoss()

    fwi_models = {n: Model(v_init_tensor, 1200.0, 4000.0).to(device) for n in OPT_NAMES}
    optimizers = {n: build_optimizer(n, fwi_models[n].parameters(), learning_rate) for n in OPT_NAMES}
    loss_histories = {n: [] for n in OPT_NAMES}

    print("Multiscale FWI — optimizers:", ", ".join(OPT_NAMES))
    print(
        f"grad_storage={grad_storage}, shots_per_batch={n_shots_per_batch}, "
        f"adamw_weight_decay={ADAMW_WEIGHT_DECAY}"
    )

    for cutoff in (10, 15, 25):
        print(f"\n--- {cutoff} Hz band ---")

        if cutoff == 10:
            lr_map = {
                "adam": 5e-3,
                "adamw": 5e-3,
                "rmsprop": 5e-3,
                "sgd": 30.0,
            }
            loss_multiplier = 1e6
        elif cutoff == 15:
            lr_map = {
                "adam": 1e-3,
                "adamw": 1e-3,
                "rmsprop": 1e-3,
                "sgd": 8.0,
            }
            loss_multiplier = 1e5
        else:
            lr_map = {
                "adam": 1e-3,
                "adamw": 1e-3,
                "rmsprop": 1e-3,
                "sgd": 2.0, 
            }
            loss_multiplier = 1e4

        for n in OPT_NAMES:
            for g in optimizers[n].param_groups:
                g["lr"] = lr_map[n]

        sos = get_filter(cutoff, dt, device)
        obs_filt_cpu = (
            apply_filt(torch.from_numpy(observed_data_cpu).to(device), sos)
            .cpu()
            .detach()
            .numpy()
        )

        for i in range(20):
            epoch_losses = {n: 0.0 for n in OPT_NAMES}

            for n in OPT_NAMES:
                optimizers[n].zero_grad(set_to_none=True)

            for batch in range(n_batch):
                batch_start = batch * n_shots_per_batch
                batch_end = min(batch_start + n_shots_per_batch, n_shots)
                sl = slice(batch_start, batch_end)
                obs_filt = torch.from_numpy(obs_filt_cpu[sl]).to(device)

                for n in OPT_NAMES:
                    v = fwi_models[n]()
                    out = scalar(
                        v,
                        (dz, dy, dx),
                        dt,
                        source_amplitudes=source_amplitudes[sl],
                        source_locations=source_locations[sl],
                        receiver_locations=receiver_locations[sl],
                        pml_freq=freq,
                        pml_width=pml_width,
                        storage_mode=grad_storage,
                    )[-1]
                    if out.device != device:
                        out = out.to(device, non_blocking=True)
                    out_filt = apply_filt(taper(out), sos)

                    batch_size = batch_end - batch_start
                    raw_loss = mse(out_filt, obs_filt) * loss_multiplier
                    loss = raw_loss * (batch_size / n_shots)
                    loss.backward()
                    epoch_losses[n] += float(raw_loss.detach().cpu()) * (batch_size / n_shots)

            for n in OPT_NAMES:
                optimizers[n].step()

            msg = " | ".join(f"{n}: {epoch_losses[n]:.4f}" for n in OPT_NAMES)
            print(f"  step {i:2d} | {msg}")
            for n in OPT_NAMES:
                loss_histories[n].append(epoch_losses[n])

    v_final = {}
    with torch.no_grad():
        for n in OPT_NAMES:
            v_final[n] = fwi_models[n]().detach().cpu().numpy()

    torch.cuda.empty_cache()

    save_dict = {
        "v_init": v_init,
        "v_true": v_true,
        "v_final_adam": v_final["adam"],
        "v_final_adamw": v_final["adamw"],
        "v_final_rmsprop": v_final["rmsprop"],
        "v_final_sgd": v_final["sgd"],
    }
    for n in OPT_NAMES:
        save_dict[f"loss_{n}"] = np.asarray(loss_histories[n], dtype=np.float64)

    np.savez(args.output, **save_dict)
    print(f"Saved {args.output.resolve()}")

    if not args.no_plot:
        plot_optimizer_comparison(
            v_true,
            v_init,
            v_final["adam"],
            v_final["adamw"],
            v_final["rmsprop"],
            v_final["sgd"],
            args.figure,
            show=args.show_plot,
        )

    print(f"\n{'':12s}  {'MAE':>10s}  {'RMSE':>10s}")
    print(
        f"{'initial':12s}  "
        f"{np.abs(v_init - v_true).mean():10.2f}  "
        f"{np.sqrt(np.mean((v_init - v_true) ** 2)):10.2f}"
    )
    for n in OPT_NAMES:
        vf = v_final[n]
        print(
            f"{n:12s}  "
            f"{np.abs(vf - v_true).mean():10.2f}  "
            f"{np.sqrt(np.mean((vf - v_true) ** 2)):10.2f}"
        )


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "plot":
        # Standalone: python test_optimizers.py plot [npz] [figure.pdf]
        _npz = Path(_sys.argv[2]) if len(_sys.argv) > 2 else Path("fwi_optimizers.npz")
        _fig = Path(_sys.argv[3]) if len(_sys.argv) > 3 else Path("optimizers-fwi.pdf")
        plot_from_npz(_npz, _fig, show=True)
    else:
        main()
