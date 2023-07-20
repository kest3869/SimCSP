# libraries 
import os
import logging
import argparse
import sys
import torch
import datetime
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Create the argument parser
parser = argparse.ArgumentParser(description='Get_boxplot')
parser.add_argument('-p', '--out_dir', type=str, help='The save path of the plots')
# Parse the command line arguments
args = parser.parse_args()
# Retrieve the values of the command line arguments
OUT_DIR = args.out_dir

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(OUT_DIR + '/results/' + 'eval_plot.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# skip if already completed 
if os.path.exists(OUT_DIR + '/results/' + '/finished_plot.pt'):
    logger.info("Found finished, skipping plot.")
    print("Found finished, skipping eval_plot.py!")
    sys.exit()

# load csv's into df's
results_path = OUT_DIR + '/results/F1_results_.csv'
paths_path = OUT_DIR + '/results/F1_paths_.csv'
results = pd.read_csv(results_path, header=None)
paths = pd.read_csv(paths_path, header=None)

# arguments 
patience = 5 # the patience used when fine-tuning the model
best_model_idx = np.shape(results)[1] - patience

# make boxplot
plt.boxplot(results.iloc[:,best_model_idx])
# add title and label
plt.title("Best Model Found at Epoch:" + str(best_model_idx))
plt.ylabel("F1 Score")
# save fig for viewing
plt.savefig(OUT_DIR + '/results/' + "F1_score.png")

# save metadata
metadata = {
    "results_path" : results_path,
    "paths_path" : paths_path,
    "patience" : patience,
    "best_model_idx" : best_model_idx,
    'paths_to_best_models' : paths.iloc[:,best_model_idx].tolist(),
    "OUT_DIR" : OUT_DIR + '/results/' + "F1_score.png"
}

# log metadata
logger.info(metadata)

# mark evaluation as finished
torch.save(datetime.datetime.now().time(), OUT_DIR + '/results/' + 'finished_plot.pt')
