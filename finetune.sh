#!/bin/bash

# the base directory for the experiment
OUT_DIR="/storage/store/kevin/local_files/exp5/"
mkdir -p "${OUT_DIR}/results/"

# path to the splits for this experiment
SPLIT_DIR="${OUT_DIR}/SPLITS/"

# BEST_SCCS
MODEL_DIR_1="/storage/store/kevin/local_files/exp5/pretrained_models/checkpoints/8500/"

# BASELINE
MODEL_DIR_2="/storage/store/kevin/data/SpliceBERT-human.510nt/"

# BEST_NMI
MODEL_DIR_3="/storage/store/kevin/local_files/exp5/pretrained_models/checkpoints/1000/"

# BEST_SCCS
OUT_DIR_1="${OUT_DIR}/BEST_SCCS/"
echo "Running finetune.sh at ${OUT_DIR_1} for model ${MODEL_DIR_1}"
# make a results directory if it does not already exist
mkdir -p "${OUT_DIR_1}/finetuned_models/"
echo "Starting fine-tuning!"
# fine-tune the model
python train.py --out-dir "${OUT_DIR_1}" --pretrained-model "${MODEL_DIR_1}" --split_dir "${SPLIT_DIR}"
echo "Finished fine-tuning!"
# generate embeddings for pre-train
python embedding_gen.py --out_dir "${OUT_DIR_1}" --only_get_last_layer
echo "Finished generating embeddings!"

# BASELINE
OUT_DIR_2="${OUT_DIR}/BASELINE/"
echo "Running finetune.sh at ${OUT_DIR_2} for model ${MODEL_DIR_2}"
# make a results directory if it does not already exist
mkdir -p "${OUT_DIR_2}/finetuned_models/"
echo "Starting fine-tuning!"
# fine-tune the model
python train.py --out-dir "${OUT_DIR_2}" --pretrained-model "${MODEL_DIR_2}" --split_dir "${SPLIT_DIR}"
echo "Finished fine-tuning!"
# generate embeddings for pre-train
python embedding_gen.py --out_dir "${OUT_DIR_2}" --only_get_last_layer
echo "Finished generating embeddings!"

# BEST_NMI
OUT_DIR_3="${OUT_DIR}/BEST_NMI/"
echo "Running finetune.sh at ${OUT_DIR_3} for model ${MODEL_DIR_3}"
# make a results directory if it does not already exist
mkdir -p "${OUT_DIR_3}/finetuned_models/"
echo "Starting fine-tuning!"
# fine-tune the model
python train.py --out-dir "${OUT_DIR_3}" --pretrained-model "${MODEL_DIR_3}" --split_dir "${SPLIT_DIR}"
echo "Finished fine-tuning!"
# generate embeddings for pre-train
python embedding_gen.py --out_dir "${OUT_DIR_3}" --only_get_last_layer
echo "Finished generating embeddings!"

# evalute the fine-tuned checkpoints
OUT_DIRS=(
  "${OUT_DIR}/BEST_SCCS/"
  "${OUT_DIR}/BASELINE/"
  "${OUT_DIR}/BEST_NMI/"
)

# loop through each OUT_DIR
for OUT_DIR_TEMP in "${OUT_DIRS[@]}"; do
  echo "Running eval.sh at ${OUT_DIR_TEMP}"

  # Make a results directory if it does not already exist
  mkdir -p "${OUT_DIR_TEMP}/results/"

  # Generate F1 scores for fine-tuned models 
  echo "Generating F1 scores!"
  python eval_F1.py --out_dir "${OUT_DIR_TEMP}" --split_dir "${SPLIT_DIR}"
  echo "Finished generating F1 scores!"

  # Make a boxplot of the best model's performance on the test data
  echo "Generating boxplot!"
  python eval_boxplot.py --out_dir "${OUT_DIR_TEMP}"
  echo "Finished generating boxplot!"

  # generate results for SCCS 
  echo "Getting SCCS Results"
  #python eval_SCCS.py --out_dir "${OUT_DIR_TEMP}" --split_dir "${SPLIT_DIR}"
  echo "Done getting SCCS Results!"

  # generate results for NMI 
  echo "Getting NMI Results"
  python eval_NMI.py --out_dir "${OUT_DIR_TEMP}" --layer "Last"
  echo "Done getting NMI results"

  current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
  echo "${OUT_DIR_TEMP} completed at time: ${current_datetime}"
done

# generate a comparative boxplot between the three models
python eval_compare.py --out_dir "${OUT_DIR}"
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
echo "finetune.sh completed at time: ${current_datetime}"
