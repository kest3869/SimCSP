#!/bin/bash

# THESE STAY CONSTANT FOR EACH EXPERIMENT
OUT_DIR="/storage/store/kevin/local_files/exp1/"
SPLIT_DIR="${OUT_DIR}/SPLITS/"

# SCCS/NMI/BASELINE model paths 
MODEL_DIR1="/storage/store/kevin/local_files/exp1/pretrained_models/checkpoints/13500/"

# specifiy the home directory
OUT_DIR_1="${OUT_DIR}/BEST_SCCS/"
echo "Running finetune.sh at ${OUT_DIR_1} for model ${MODEL_DIR1}"
# make a results directory if it does not already exist
mkdir -p "${OUT_DIR_1}/finetuned_models/"
echo "Starting fine-tuning!"
# fine-tune the model
python train.py --out-dir "${OUT_DIR_1}" --pretrained-model "${MODEL_DIR1}" --split_dir "${SPLIT_DIR}"
echo "Finished fine-tuning! Generating Embeddings"

# evalute the fine-tuned checkpoints
OUT_DIRS2=(
  "${OUT_DIR}/BEST_SCCS/"
)

# adjust OUT_DIRS# based on usage 
for OUT_DIR_TEMP in "${OUT_DIRS2[@]}"; do
  echo "Running eval.sh at ${OUT_DIR}"

  # Make a results directory if it does not already exist
  mkdir -p "${OUT_DIR_TEMP}/results/"

  # Generate SCCS data
  echo "Generating SCCS data!"
  python eval_SCCS.py --out_dir "${OUT_DIR_TEMP}" --split_dir "${SPLIT_DIR}"
  echo "Finished generating SCCS data!"

  # Generate F1 scores for fine-tuned models 
  echo "Generating F1 scores!"
  python eval_F1.py --out_dir "${OUT_DIR_TEMP}" --split_dir "${SPLIT_DIR}"
  echo "Finished generating F1 scores!"

  # Make a boxplot of the best model's performance on the test data
  echo "Generating boxplot!"
  python eval_boxplot.py --out_dir "${OUT_DIR_TEMP}"
  echo "Finished generating boxplot!"

  current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
  echo "${OUT_DIR_TEMP} completed at time: ${current_datetime}"
done

echo "Generating comparative plot"
python eval_compare.py --out_dir "${OUT_DIR}"
echo "Comparative plot generated" 

current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
echo "finetune.sh completed at time: ${current_datetime}"