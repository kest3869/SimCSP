#!/bin/bash

# specifiy the home directory
OUT_DIR="/storage/store/kevin/local_files/scrap/scrap_exp1/"

# make the directory if it does not exist 
mkdir -p "$OUT_DIR"
echo "Running fit at $OUT_DIR"

# make the sub-directories if they don't already exist 
mkdir -p "${OUT_DIR}/pretrained_models/"
mkdir -p "${OUT_DIR}/finetuned_models/"

# pre-train the model
echo "Starting pre-training!"
python pretrain.py --model_save_path "${OUT_DIR}"
echo "Finished pre-training! Starting fine-tuning!"

# fine-tune the model
python train.py --out-dir "${OUT_DIR}" --pretrained-model "${OUT_DIR}"
echo "Finished fine-tuning! Generating Embeddings"

# generate embeddings for pre-train
python embedding_gen.py --out_dir "${OUT_DIR}" --only_get_last_layer
echo "Finished generating embeddings!"