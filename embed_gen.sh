#!/bin/bash

# Set the arguments
bed="/home/data/hg19.ss-motif.for_umap.bed.gz"
root_dir="/home/search_space/hrg_val_test/512/3e5/embed/pretrain"

# initialize logger in root dir 
logger_file="${root_dir}/logger_gen.txt"
exec > >(tee -a "$logger_file") 2>&1

# numbers to generate embeddings for 
numbers=(6591 8112 10140 12168 12700)

for num in "${numbers[@]}"; do
    model="/home/search_space/hrg_val_test/512/3e5/pretrained/${num}/"
    out_dir="${root_dir}/${num}/gen/"
    mkdir -p "$out_dir"
    echo $out_dir
    python embedding_gen.py --model "$model" --bed "$bed" --out_dir "$out_dir"
done