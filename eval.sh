#!/bin/bash

# specifiy the home directory
OUT_DIR="/storage/store/kevin/local_files/exp1/"
echo "Running eval.sh at $OUT_DIR"

# make a results directory if it does not already exist
mkdir -p "${OUT_DIR}/results/"

# generate NMI data
echo "Generating NMI data!" 
python eval_NMI.py --out_dir "${OUT_DIR}" --layer "last"
echo "Finished generating NMI data!"

# generate SCCS data
echo "Generating SCCS data!"
python eval_SCCS.py --out_dir "${OUT_DIR}"
echo "Finished generating SCCS dats!"