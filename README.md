# GeoFWI3D
[![arXiv](https://img.shields.io/badge/arXiv-2610.01033-b31b1b.svg)](https://arxiv.org/abs/2610.01033)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-1682D4.svg)](https://doi.org/10.5281/zenodo.20148778)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20148778.svg)](https://doi.org/10.5281/zenodo.20148778)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Large-scale 3D velocity models for deep-learning full waveform inversion (FWI) and other seismic processing workflows.

Four representative models from the GeoFWI3D dataset

The accompanying article is available on [arXiv](https://arxiv.org/abs/2610.01033).

# Citation

If you use the **GeoFWI3D** dataset in your research, please cite the accompanying article:

```bibtex
@misc{geofwi3d,
      title={GeoFWI3D: Large-scale 3D Velocity Model Dataset for Deep Learning-assisted Seismic Imaging}, 
      author={Sujith Swaminadhan and Yiran Shen and Chao Li and Kai Gao and Ting Chen and Sergey Fomel and Tolulope Agbaje and Yang Cui and Liuqing Yang and Jaewook Lee and Robin Dommisse and Umair bin Waheed and Mrinal K. Sen and Yangkang Chen},
      year={2026},
      eprint={2610.01033},
      archivePrefix={arXiv},
      primaryClass={physics.geo-ph},
      url={https://arxiv.org/abs/2610.01033}, 
}
```

The dataset can also be cited:

```bibtex
@misc{geofwi3d-dataset,
  author    = {Swaminadhan, Sujith and Shen, Yiran and Li, Chao and Gao, Kai and Chen, Ting and Fomel, Sergey and Agbaje, Tolulope and Cui, Yang and Yang, Liuqing and Lee, Jaewook and Dommisse, Robin and Waheed, Umair bin and Sen, Mrinal K. and Chen, Yangkang},
  title     = {GeoFWI3D: Large-scale 3D Velocity Model Dataset for Deep Learning-assisted Seismic Imaging},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20148778},
  url       = {https://doi.org/10.5281/zenodo.20148778},
  note      = {Dataset}
}
```

**Dataset:** [GeoFWI3D: Large-scale 3D Velocity Model Dataset for Deep Learning-assisted Seismic Imaging](https://doi.org/10.5281/zenodo.20148778)

## Overview

**GeoFWI3D** is a large-scale dataset of synthetic, geologically diverse 3D
subsurface models designed for deep-learning-assisted full waveform inversion
(FWI), seismic imaging, and related seismic processing tasks.

The primary dataset contains **10,000 models with dimensions 96 × 96 × 96**.
An additional collection of higher-resolution **256 × 256 × 256** models is
also provided.


| File          | Contents                     |
| ------------- | ---------------------------- |
| `vp3d.bin`    | P-wave velocity              |
| `image3d.bin` | Synthetic p-reflectivity     |
| `rgt3d.bin`   | Relative geologic time (RGT) |
| `fault3d.bin` | Fault index mask             |


The binary arrays are stored contiguously in C order.

## Download

The dataset is available through both Zenodo and Box.

- [Zenodo](https://doi.org/10.5281/zenodo.20148778)
- [Box download](https://utexas.box.com/s/ybzgil0u3hvgusoc27bechxibyck88jr)

### 96 × 96 × 96 models

The primary dataset with **10,000** models is distributed as:

```text
models_batch_*.tar.gz
```

Each archive contains **1,000** models.

### 256 × 256 × 256 models

The high-resolution dataset with **2,000** models are distributed as:

```text
models_256_batch_*.tar.gz
```

Each archive contains **400** models.

## Extract

From the repository root:

```bash
./extract_models.sh
```

By default this creates `allmodels/` and extracts every 96 x 96 x96 models into it. To extract large models uncomment the corresponding lines at the bottom.

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

allmodels_256/
├── model_0000/
│   ├── image3d.bin
│   ├── vp3d.bin
│   ├── rgt3d.bin
│   └── fault3d.bin
└── ...
```

Folder names use four-digit zero padding: `model_0000`, `model_0001`, …

## Quick start

1. Install Python dependencies used by the example notebook: NumPy, Matplotlib, and scikit-image (for `marching_cubes`).
2. Run Jupyter with working directory `[quick_start/](quick_start/)` so `from plotting import plot3d` works. Open `[read_data.ipynb](quick_start/read_data.ipynb)`: it defines `read_models` and `plot_all_models`, sets `data_root` / `model_folders` / `shape`, and walks through loading and plotting.
3. For large models set the shape to `shape = (256, 256, 256)` and use `parameters_256.csv`

### Model categories from `parameters.csv`

The repository includes 

- `[parameters.csv](parameters.csv)` - 96 × 96 × 96 models
- `[parameters_256.csv](parameters_256.csv)` - 256 × 256 × 256 models

Both files contain key `sample_index` column mapping folder names such as `model_{sample_index:04d}`.

Useful label columns for categorization:

- `yn_fault`: 1 if model contains faults
- `yn_salt`: 1 if model contains a salt body

Example (from `quick_start/read_data.ipynb`) to build category index lists and select models:

```python
import csv
from pathlib import Path

with Path("../parameters.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r["sample_index"] = int(r["sample_index"])
    r["yn_fault"] = int(float(r["yn_fault"]))
    r["yn_salt"] = int(float(r["yn_salt"]))

def select_indices(rows, **flag_equals):
    return [
        r["sample_index"]
        for r in rows
        if all(int(r[k]) == int(v) for k, v in flag_equals.items())
    ]

layered_indices = select_indices(rows, yn_fault=0, yn_salt=0)
fault_indices = select_indices(rows, yn_fault=1, yn_salt=0)
salt_indices = select_indices(rows, yn_salt=1, yn_fault=0)
fault_and_salt_indices = select_indices(rows, yn_fault=1, yn_salt=1)

idx = salt_indices[0]
image, vp, rgt, fault, salt = read_models(data_root, model_folders, idx)
```

### Salt + Fault model

```python
plot_all_models(
    data_root, 
    model_folders, 
    9177, 
    "Salt model with faults",
    save_path="../gallery/salt_fault_model.png"
)
```

Model 9177 — seismic image, velocity, RGT, and fault models

### Fault mask

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

### Salt mask

Fault index 5 mask (same model as above)

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

Salt mask

3D salt mask


## License

The GeoFWI3D dataset is released under the

[Creative Commons Attribution 4.0 International (CC BY 4.0)]([https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)).

---

*Plotting helpers in `quick_start/plotting.py` are adapted from pyseistr ([https://github.com/aaspip/pyseistr](https://github.com/aaspip/pyseistr)) utilities.*