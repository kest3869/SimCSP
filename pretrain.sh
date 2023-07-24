#!/bin/bash

# THESE STAY CONSTANT FOR EACH EXPERIMENT
OUT_DIR_BASE="/storage/store/kevin/local_files/exp3/"
SPLIT_DIR="${OUT_DIR_BASE}/SPLITS/"
PRETRAINED_MODELS="/storage/store/kevin/data/chopped_models/SpliceBERT-human.510nt-chop_"

# making necessary directories
mkdir -p "$SPLIT_DIR"
echo "Made SPLIT_DIR at ${SPLIT_DIR}"
python split_spliceator.py --split_dir "${SPLIT_DIR}"
echo "Finished splitting! Splits saved at ${SPLIT_DIR}"
mkdir -p "$OUT_DIR_BASE"
echo "Running pretrain.sh"

# args
bs=512
lr=0.0001
wd=0.000001

# hyperparameters to search
chops=(1 2)

# performs hyper-param search
for chop in "${chops[@]}"; do
  OUT_DIR="${OUT_DIR_BASE}/chop-${chop}/"
  mkdir -p "${OUT_DIR}/pretrained_models/"
  echo "Starting pre-training!"

  python pretrain.py \
    --model_save_path "${OUT_DIR}" \
    --pretrained_model_path "${PRETRAINED_MODELS}${chop}/" \
    --split_dir "${SPLIT_DIR}" \
    --batch_size "${bs}" \
    --learning_rate "${lr}" \
    --weight_decay "${wd}"

  current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
  echo "pretrain.py for ${OUT_DIR} completed at time: ${current_datetime}!"
done

# displays finish time for script
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
echo "pretrain.sh for ${OUT_DIR_BASE} completed at time: ${current_datetime}!"
