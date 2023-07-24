#!/bin/bash

# THESE STAY CONSTANT FOR EACH EXPERIMENT 
OUT_DIR_BASE="/storage/store/kevin/local_files/exp2/"
SPLIT_DIR="${OUT_DIR_BASE}/SPLITS/"

# making necessary directories
mkdir -p "$SPLIT_DIR"
echo "Made SPLIT_DIR at ${SPLIT_DIR}"
python split_spliceator.py --split_dir "${SPLIT_DIR}"
echo "Finished splitting! Splits saved at ${SPLIT_DIR}"
mkdir -p "$OUT_DIR_BASE"
echo "Running pretrain.sh"

# hyperparameters to search
batch_sizes=(512)
learning_rates=(0.0001)
weight_decays=(0.000001)

# performs hyper-param search 
for bs in "${batch_sizes[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for wd in "${weight_decays[@]}"; do
      OUT_DIR="${OUT_DIR_BASE}/${bs}+${lr}+${wd}"
      mkdir -p "${OUT_DIR}/pretrained_models/"
      echo "Starting pre-training!"
      python pretrain.py --model_save_path "${OUT_DIR}" --split_dir $SPLIT_DIR --batch_size $bs --learning_rate $lr --weight_decay $wd 
      current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
      echo "pretrain.py for ${OUT_DIR} completed at time: ${current_datetime}!"
    done
  done
done

# displays finish time for script
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
echo "pretrain.sh for ${OUT_DIR_BASE} completed at time: ${current_datetime}!"