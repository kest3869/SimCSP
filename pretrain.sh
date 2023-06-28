#!/bin/bash

# Create parent directory if it doesn't exist
parent_dir="/home/search_space/hrg_train/"
mkdir -p "$parent_dir"

# Construct OUT_DIR
OUT_DIR="${parent_dir}/512/3e5/"

# Create directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Log file path
log_file="${parent_dir}/output.log"

# Pretrain a model with the associated learning rate and batch size 
python pretrain.py --model_save_path "$OUT_DIR" > "$log_file" 2>&1
