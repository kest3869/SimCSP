#!/bin/bash
OUT_DIR="/storage/store/kevin/local_files/scrap/exp1/"
SPLIT_DIR='/storage/store/kevin/local_files/exp1/SPLITS/'
python eval_F1.py --out_dir "${OUT_DIR}" --split_dir "${SPLIT_DIR}"