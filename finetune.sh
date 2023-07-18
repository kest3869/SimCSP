#!/bin/bash

# specifiy the home directory
OUT_DIR="/storage/store/kevin/local_files/exp1/BEST_NMI/"
MODEL_DIR="/storage/store/kevin/local_files/exp1/pretrained_models/checkpoints/2000/"
echo "Running finetune.sh at $OUT_DIR for model $MODEL_DIR!"

# make a results directory if it does not already exist
mkdir -p "${OUT_DIR}/finetuned_models/"
echo "Starting fine-tuning!"

# fine-tune the model
python train.py --out-dir "${OUT_DIR}" --pretrained-model "${MODEL_DIR}"
echo "Finished fine-tuning! Generating Embeddings"

# generate embeddings for pre-train
python embedding_gen.py --out_dir "${OUT_DIR}" --only_get_last_layer
echo "Finished generating embeddings!"