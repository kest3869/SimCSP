#!/bin/bash

# the base directory for the experiment
OUT_DIR="/storage/store/kevin/local_files/exp5/"
mkdir -p "${OUT_DIR}/results/"

# path to the splits for this experiment
SPLIT_DIR="${OUT_DIR}/SPLITS/"

# evalute the fine-tuned checkpoints
OUT_DIRS=(
  "${OUT_DIR}/BEST_SCCS/"
  "${OUT_DIR}/BASELINE/"
  "${OUT_DIR}/BEST_NMI/"
)

# loop through each OUT_DIR
for OUT_DIR_TEMP in "${OUT_DIRS[@]}"; do

  # Make a results directory if it does not already exist
  mkdir -p "${OUT_DIR_TEMP}/results/"
  python eval_SCCS.py --out_dir "${OUT_DIR_TEMP}" --split_dir "${SPLIT_DIR}"

done

