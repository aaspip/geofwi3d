#!/bin/bash

mkdir -p allmodels

for archive in models_batch_*.tar.gz; do
    echo "Extracting $archive into allmodels/..."
    tar -xzf "$archive" -C allmodels
done


## uncomment to extract large models (256x256x256)

# mkdir -p allmodels_256
# for archive in models_256_batch_*.tar.gz; do
#     echo "Extracting $archive into allmodels_256/..."
#     tar -xzf "$archive" -C allmodels_256
# done