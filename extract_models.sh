#!/bin/bash

mkdir -p allmodels

for archive in models_batch_*.tar.gz; do
    echo "Extracting $archive into allmodels/..."
    tar -xzf "$archive" -C allmodels
done
