# GeoFWI3D

Large-scale 3D velocity models for deep-learning full waveform inversion (FWI) and other seismic processing workflows.

![Four representative models from the GeoFWI3D dataset](./gallery/geofwi3d_models.png)

## Overview

**GeoFWI3D** provides many synthetic 3D Earth models, each stored as raw `float32` (little-endian) binaries. Per model you get:

| File | Contents |
|------|----------|
| `vp3d.bin` | P-wave velocity |
| `image3d.bin` | Seismic image |
| `rgt3d.bin` | Relative geologic time (RGT) |
| `fault3d.bin` | Fault index mask |

Each volume has shape **96 × 96 × 96** in **(X, Y, Z)** order. On disk the layout is contiguous C-order over those dimensions.

## Download

Compressed archives are hosted on Box:

[Download GeoFWI3D data](https://utexas.box.com/s/ybzgil0u3hvgusoc27bechxibyck88jr)

Each `models_batch_*.tar.gz` contains **1000** models. Place the archive files in the same directory as this repository, then extract.

## Extract

From the repository root:

```bash
./extract_models.sh
```

This creates `allmodels/` and extracts every `models_batch_*.tar.gz` into it.

## Directory layout

After extraction, models look like:

```text
allmodels/
├── model_0000/
│   ├── image3d.bin
│   ├── vp3d.bin
│   ├── rgt3d.bin
│   └── fault3d.bin
├── model_0001/
│   └── ...
└── ...
```

Folder names use four-digit zero padding: `model_0000`, `model_0001`, …

## Quick start

1. Install Python dependencies used by the example notebook: NumPy, Matplotlib, and scikit-image (for `marching_cubes`).
2. Run Jupyter with working directory [`quick_start/`](quick_start/) so `from plotting import plot3d` works. Open [`read_data.ipynb`](quick_start/read_data.ipynb): it defines `read_models` and `plot_all_models`, sets `data_root` / `model_folders` / `shape`, and walks through loading and plotting.

```python
plot_all_models(
    data_root, 
    model_folders, 
    9177, 
    "Salt model with faults",
    save_path="../gallery/salt_fault_model.png"
)
```

![Model 9177 — seismic image, velocity, RGT, and fault models](./gallery/salt_fault_model.png)

`fault3d.bin` stores a **fault index** per voxel. To plot a single fault (here index `5`):

```python
image, vp, rgt, fault, salt = read_models(data_root, model_folders, 9177)
fault_mask = fault.T == 5
plot3d(
    fault_mask.astype(np.float32),
    cmap='Reds',
    frames=[45, 45, 46],
    ifnewfig=True,  
    showf=False,     
    close=False,     
    ifinside=False,
    figname="../gallery/fault_mask_5.png"
)
plt.show()
```

![Fault index 5 mask (same model as above)](./gallery/fault_mask_5.png)

Salt bodies have **RGT = 0** in `rgt3d.bin`. Mask and plot with:

```python
salt_mask = rgt == 0
salt_mask = salt_mask.astype(np.float32)
plot3d(
    salt_mask.T,
    cmap='Reds',
    frames=[45, 45, 46],
    ifnewfig=True,   
    figname="../gallery/salt_mask.png"
)
plt.show()
```

![Salt mask](./gallery/salt_mask.png)

![3D salt mask](./gallery/salt_body_rotation.gif)

---

*Plotting helpers in `quick_start/plotting.py` are adapted from pyseistr (https://github.com/aaspip/pyseistr) utilities.*
