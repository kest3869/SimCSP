#!/bin/bash

# Log file path
log_file="/home/search_log.log"

# Redirect all terminal output to the log file
exec > >(tee -a "$log_file") 2>&1

# Search_space
bss=(64 256 512) # batch sizes
lrs=(1e-5 3e-5 5e-5) # learning rates

# Converts hyper-parameter to string associated with correct directory
declare -A hp_str
hp_str[64]='64'
hp_str[256]='256'
hp_str[512]='512'
hp_str[1e-5]='1e5'
hp_str[3e-5]='3e5'
hp_str[5e-5]='5e5'

# Create parent directory if it doesn't exist
parent_dir="/home/search_space/hrg_validation/"
mkdir -p "$parent_dir"

# Search loop
for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do

        # Construct OUT_DIR
        OUT_DIR="${parent_dir}${hp_str[$bs]}/${hp_str[$lr]}/"
        
        # Create directory if it doesn't exist
        mkdir -p "$OUT_DIR"
        
        # Pretrain a model with the associated learning rate and batch size 
        python search.py --batch_size "$bs" --learning_rate "$lr" --model_save_path "$OUT_DIR"
    done
done

