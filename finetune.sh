#!/bin/bash

# THESE STAY CONSTANT FOR EACH EXPERIMENT
OUT_DIR="/storage/store/kevin/local_files/exp1/"
SPLIT_DIR="${OUT_DIR}/SPLITS/"

# SCCS/NMI/BASELINE model paths 
MODEL_DIR1="/storage/store/kevin/local_files/exp1/pretrained_models/checkpoints/6000"
MODEL_DIR2="/storage/store/kevin/local_files/exp1/pretrained_models/checkpoints/1500/"
MODEL_DIR3="/storage/store/kevin/data/SpliceBERT-human.510nt"

# specifiy the home directory
OUT_DIR_1="${OUT_DIR}/BEST_SCCS/"
echo "Running finetune.sh at ${OUT_DIR_1} for model ${MODEL_DIR1}"
# make a results directory if it does not already exist
mkdir -p "${OUT_DIR_1}/finetuned_models/"
echo "Starting fine-tuning!"
# fine-tune the model
python train.py --out-dir "${OUT_DIR_1}" --pretrained-model "${MODEL_DIR1}" --split_dir "${SPLIT_DIR}"
echo "Finished fine-tuning! Generating Embeddings"
# generate embeddings for pre-train
python embedding_gen.py --out_dir "${OUT_DIR_1}" --only_get_last_layer
echo "Finished generating embeddings!"

# specifiy the home directory
OUT_DIR_2="${OUT_DIR}/BEST_NMI/"
echo "Running finetune.sh at ${OUT_DIR_2} for model ${MODEL_DIR2}"
# make a results directory if it does not already exist
mkdir -p "${OUT_DIR_2}/finetuned_models/"
echo "Starting fine-tuning!"
# fine-tune the model
python train.py --out-dir "${OUT_DIR_2}" --pretrained-model "${MODEL_DIR2}" --split_dir "${SPLIT_DIR}"
echo "Finished fine-tuning! Generating Embeddings"
# generate embeddings for pre-train
python embedding_gen.py --out_dir "${OUT_DIR_2}" --only_get_last_layer
echo "Finished generating embeddings!"

# specifiy the home directory
OUT_DIR_3="${OUT_DIR}/BASELINE/"
echo "Running finetune.sh at ${OUT_DIR_3} for model ${MODEL_DIR}"
# make a results directory if it does not already exist
mkdir -p "${OUT_DIR_3}/finetuned_models/"
echo "Starting fine-tuning!"
# fine-tune the model
python train.py --out-dir "${OUT_DIR_3}" --pretrained-model "${MODEL_DIR3}" --split_dir "${SPLIT_DIR}"
echo "Finished fine-tuning! Generating Embeddings"
# generate embeddings for pre-train
python embedding_gen.py --out_dir "${OUT_DIR_3}" --only_get_last_layer
echo "Finished generating embeddings!"

# evalute the fine-tuned checkpoints
OUT_DIRS2=(
  "${OUT_DIR}/BEST_SCCS/"
  "${OUT_DIR}/BEST_NMI/"
  "${OUT_DIR}/BASELINE/"
)

# adjust OUT_DIRS# based on usage 
for OUT_DIR_TEMP in "${OUT_DIRS2[@]}"; do
  echo "Running eval.sh at ${OUT_DIR}"

  # Make a results directory if it does not already exist
  mkdir -p "${OUT_DIR_TEMP}/results/"

  # Generate NMI data
  echo "Generating NMI data!" 
  python eval_NMI.py --out_dir "${OUT_DIR_TEMP}" --layer "last"
  echo "Finished generating NMI data!"

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