#!/bin/bash

# Set the arguments
bed="/home/data/hg19.ss-motif.for_umap.bed.gz"
root_dir="/home/search_space/hrg_val_test/512/3e5/embed/pretrain"

# initialize logger in root dir 
logger_file="${root_dir}/logger_plot.txt"
exec > >(tee -a "$logger_file") 2>&1

# numbers to generate embeddings for 
numbers=(507 2535 4563 6591 8112 10140 12168 12700)
layers=(0 1 2 3 4 5 6)

for num in "${numbers[@]}"; do
    for layer in "${layers[@]}"; do
        data="/home/search_space/hrg_val_test/512/3e5/embed/pretrain/${num}/gen/hg19.ss-motif.for_umap.bed.gz.L${layer}.h5ad"
        out_dir="${root_dir}/${num}/plots/"
        mkdir -p "$out_dir"
        echo "$out_dir"
        python embedding_plot.py --data "$data" --out_dir "$out_dir" --layer "$layer"
    done
done
