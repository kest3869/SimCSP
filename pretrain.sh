#!/bin/bash

# THESE STAY CONSTANT FOR EACH EXPERIMENT 
OUT_DIR="/storage/store/kevin/local_files/exp1/"
SPLIT_DIR="${OUT_DIR}/SPLITS/"

# make OUT_DIR if it does not exist 
mkdir -p "$OUT_DIR"
echo "Running fit.sh at $OUT_DIR"

# make SPLIT_DIR if it does not exist
mkdir -p "$SPLIT_DIR"
echo "Made SPLIT_DIR at ${SPLIT_DIR}"

# make the sub-directories if they don't already exist 
mkdir -p "${OUT_DIR}/pretrained_models/"
mkdir -p "${OUT_DIR}/finetuned_models/" 

# pre-train the model
echo "Starting pre-training!"
python pretrain.py --model_save_path "${OUT_DIR}"
echo "Finished pre-training! Making splits!"

# Generate split for this experiement
python split_spliceator.py --split_dir "${SPLIT_DIR}"
echo "Finished splitting! Splits saved at ${SPLIT_DIR}"

# generate embeddings for pre-train
echo "Generating embeddings!"
python embedding_gen.py --out_dir "${OUT_DIR}" --only_get_last_layer
echo "Finished generating embeddings!"

# print the time completed 
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
echo "pretrain.sh completed at time: ${current_datetime}!"

# use this to determine which pre-trained checkpoints should be used for fine-tuning
OUT_DIRS1=("${OUT_DIR}")

# adjust OUT_DIRS# based on usage 
for OUT_DIR in "${OUT_DIRS1[@]}"; do
  echo "Running eval.sh at ${OUT_DIR}"

  # Make a results directory if it does not already exist
  mkdir -p "${OUT_DIR}/results/"

  # Generate NMI data
  echo "Generating NMI data!" 
  python eval_NMI.py --out_dir "${OUT_DIR}" --layer "last"
  echo "Finished generating NMI data!"

  # Generate SCCS data
  echo "Generating SCCS data!"
  python eval_SCCS.py --out_dir "${OUT_DIR}" --split_dir "${SPLIT_DIR}"
  echo "Finished generating SCCS data!"

  current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
  echo "eval.sh completed at time: ${current_datetime} for ${OUT_DIR}!"
done
