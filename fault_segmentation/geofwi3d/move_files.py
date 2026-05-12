import pandas as pd
params = pd.read_csv(".../../parameters.csv")

idx_for_fault_only = params[
    (params["yn_salt"] == 0) &
    (params["yn_fault"] == 1) &
    (params["unconf"] == 0) & 
    (params["nf"] > 8)
].index

ind = idx_for_fault_only[:200]
print(ind)

import os, shutil

# assuming from quick_start/read_data.ipynb the downloaded data is in the ../allmodels folder
data_root = "../allmodels/"

for i, idx in enumerate(ind):
    print(f'Copying {idx} to {i}')
    shutil.copy(f"{data_root}/model_{idx:04d}/image3d.bin", f"./data/train/seis/{i}.dat")
    shutil.copy(f"{data_root}/model_{idx:04d}/fault3d.bin", f"./data/train/fault/{i}.dat")

ind = idx_for_fault_only[250:270]
    
for i, idx in enumerate(ind):
    print(f'Copying {idx} to {i}')
    shutil.copy(f"{data_root}/model_{idx:04d}/image3d.bin", f"./data/validation/seis/{i}.dat")
    shutil.copy(f"{data_root}/model_{idx:04d}/fault3d.bin", f"./data/validation/fault/{i}.dat")