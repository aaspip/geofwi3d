#!/bin/bash

mkdir -p allmodels

for archive in models_batch_*.tar.gz; do
    echo "Extracting $archive into allmodels/..."
    tar -xzf "$archive" -C allmodels
done


# Extract large models if the files exist
# NOTE: If you have 'pigz' installed, you can significantly speed up 
# extraction by replacing 'tar -xzf' with 'tar -I pigz -xf' in the loops below.

if ls models_256_batch_*.tar.gz >/dev/null 2>&1; then
    mkdir -p allmodels_256
    for archive in models_256_batch_*.tar.gz; do
        echo "Extracting $archive into allmodels_256/..."
        tar -xzf "$archive" -C allmodels_256          # comment this if using pigz
        # tar -I pigz -xf "$archive" -C allmodels_256 # uncomment to use pigz for faster extraction
    done
fi