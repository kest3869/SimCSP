#!/bin/bash

# THESE STAY CONSTANT FOR EACH EXPERIMENT
OUT_DIR_BASE="/storage/store/kevin/local_files/exp5/"
SPLIT_DIR="${OUT_DIR_BASE}/SPLITS/"
PRETRAINED_MODEL="/storage/store/kevin/data/SpliceBERT-human.510nt/"

# making necessary directories
mkdir -p "$SPLIT_DIR"
echo "Made SPLIT_DIR at ${SPLIT_DIR}"
python split_spliceator.py --split_dir "${SPLIT_DIR}"
echo "Finished splitting! Splits saved at ${SPLIT_DIR}"
mkdir -p "${OUT_DIR_BASE}"
mkdir -r "${OUT_DIR_BASE}/results/"
echo "Running pretrain.sh"


# args
bs=512
lr=0.0001
wd=0.000001

OUT_DIR="${OUT_DIR_BASE}"
mkdir -p "${OUT_DIR}/pretrained_models/"
echo "Starting pre-training!"

python pretrain.py \
  --model_save_path "${OUT_DIR}" \
  --pretrained_model_path "${PRETRAINED_MODEL}" \
  --split_dir "${SPLIT_DIR}" \
  --batch_size "${bs}" \
  --learning_rate "${lr}" \
  --weight_decay "${wd}"

echo "generating embeddings"
python embedding_gen.py --out_dir "${OUT_DIR_BASE}" --only_get_last_layer
echo "finished generating embeddings"

# displays finish time for script
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
echo "pretrain.sh for ${OUT_DIR_BASE} completed at time: ${current_datetime}!"
