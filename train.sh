#!/bin/bash

# Define the list of numbers
numbers=(507 2535 4563 6591 8112 10140 12168 12700)

# Create parent directory if it doesn't exist
parent_dir="/home/search_space/hrg_val_test/512/3e5"
mkdir -p "$parent_dir"

# Loop over the numbers and run train.py
for number in "${numbers[@]}"; do
    # Construct OUT_DIR with the current number
    OUT_DIR="${parent_dir}/${number}"

    # Create directory if it doesn't exist
    mkdir -p "$OUT_DIR"

    # Log file path with the number
    log_file="${OUT_DIR}/output_finetune_${number}.log"

    # Set the current number in the PRETRAINED_MODEL path
    PRETRAINED_MODEL="${parent_dir}/pretrained/$number"

    # Pretrain a model with the associated learning rate and batch size
    python train.py --out-dir "$OUT_DIR" --pretrained-model "$PRETRAINED_MODEL" > "$log_file" 2>&1
done
