#!/bin/bash

# Log file path
log_file="/home/search_log.log"

# Redirect all terminal output to the log file
exec > >(tee -a "$log_file") 2>&1

# Search_space
eps=(1 3 5) # num epochs

# Converts hyper-parameter to string associated with correct directory
declare -A hp_str
hp_str[1]='1'
hp_str[3]='3'
hp_str[5]='5'

# Create parent directory if it doesn't exist
parent_dir="/home/search_space/hrg_train/"
mkdir -p "$parent_dir"

# Search loop
for ep in "${eps[@]}"; do

    # Construct OUT_DIR
    OUT_DIR="${parent_dir}${hp_str[$ep]}/"
    
    # Create directory if it doesn't exist
    mkdir -p "$OUT_DIR"
    
    # Pretrain a model with the associated learning rate and batch size 
    python search.py --epochs "$ep"  --out_dir "$OUT_DIR"

done

