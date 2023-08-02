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
parser = argparse.ArgumentParser(description='Get_boxplot_comparison')
parser.add_argument('-p', '--out_dir', type=str, help='The save path of the plots')
# Parse the command line arguments
args = parser.parse_args()
# Retrieve the values of the command line arguments
OUT_DIR = args.out_dir

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(OUT_DIR + '/results/' + 'eval_compare.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# load csv's into df's for each version
versions = ['BASELINE', 'BEST_SCCS', 'BEST_NMI']
results_dfs = {}
paths_dfs = {}
for version in versions:
    results_path = OUT_DIR + '/' + version + '/' + '/results/F1_results_.csv'
    paths_path = OUT_DIR + '/' + version + '/' + '/results/F1_paths_.csv'
    results_df = pd.read_csv(results_path, header=None)
    paths_df = pd.read_csv(paths_path, header=None)
    results_dfs[version] = results_df
    paths_dfs[version] = paths_df

# Set up a single plot with three subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Initialize a variable to store the maximum F1 score for all versions
max_f1_score = 0.995

# Loop through each version and perform operations
for i, version in enumerate(versions):
    results_df = results_dfs[version]
    paths_df = paths_dfs[version]

    patience = 5  # the patience used when fine-tuning the model
    best_model_idx = np.shape(results_df)[1] - patience

    # Calculate mean
    mean_f1 = np.mean(results_df.iloc[:, best_model_idx])
    print(f"Mean F1 Score for {version}: {mean_f1}")

    # Make boxplot
    axes[i].boxplot(results_df.iloc[:, best_model_idx])

    # Add title and label for each subplot
    axes[i].set_title(f"Epoch: {best_model_idx}")
    axes[i].set_xlabel(str(version))
    axes[i].set_ylabel("F1 Score")

    # Update the maximum F1 score if needed
    max_f1 = np.max(results_df.iloc[:, best_model_idx])
    max_f1_score = max(max_f1_score, max_f1)

# Set the same y-axis limits for all subplots
for ax in axes:
    ax.set_ylim(0.960, max_f1_score)

# Save the entire figure with all three subplots
plt.savefig(os.path.join(OUT_DIR, 'results', "F1_score_compare.png"))

# Save metadata
metadata_combined = {}
for version in versions:
    results_df = results_dfs[version]
    paths_df = paths_dfs[version]
    patience = 5
    best_model_idx = np.shape(results_df)[1] - patience
    metadata_combined[version] = {
        "results_path": OUT_DIR + '/' + version + '/' + '/results/F1_results_.csv',
        "paths_path": OUT_DIR + '/' + version + '/' + '/results/F1_paths_.csv',
        "patience": patience,
        "best_model_idx": best_model_idx,
        'paths_to_best_models': paths_df.iloc[:, best_model_idx].tolist(),
        "OUT_DIR": os.path.join(OUT_DIR, 'results', "F1_score_combined.png")
    }

# log metadata
logger.info(metadata_combined)

# mark evaluation as finished
torch.save(datetime.datetime.now().time(), os.path.join(OUT_DIR, '/results/', '/finished_plot.pt'))