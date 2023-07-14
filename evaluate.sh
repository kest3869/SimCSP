#!/bin/bash

# specifiy the home directory
OUT_DIR="/storage/store/kevin/local_files/scrap/scrap_exp1/"

# make the directory if it does not exist 
mkdir -p "$OUT_DIR"
echo "Running evaluate at $OUT_DIR"

# generate embeddings 
mkdir -p "${OUT_DIR}/embeddings/pretrained/"
mkdir -p "${OUT_DIR}/embeddings/finetuned/"
python embedding_gen.py --model "${OUT_DIR}/pretrained_models/" --out_dir "${OUT_DIR}/embeddings/pretrained/" --only_get_last_layer
echo "Finished generating embeddings for pretrained models! Starting embeddings for fine-tuned models!"
# add a for loop here for each fold
python embedding_gen.py --model "${OUT_DIR}/finetuned_models/fold0" --out_dir "${OUT_DIR}/embeddings/finetuned/" --only_get_last_layer
echo "Finished generating embeddings for fine-tuned models!"
