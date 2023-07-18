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
parser = argparse.ArgumentParser(description='Get_Plots')
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

'''
# skip if already completed 
if os.path.exists(OUT_DIR + '/results/' + '/finished_plot.pt'):
    logger.info("Found finished, skipping plot.")
    print("Found finished, skipping eval_plot.py!")
    sys.exit()
'''

# the filenames holding our files 
file_names = ['F1_paths_', 'F1_results_',
              'finetune_NMI_results.csv', 'pretrain_NMI_results',
              'SCCS_results_finetuned', 'SCCS_results_finetuned']

paths = []
for file_name in file_names:
    paths.append(file_name + '.csv') # TODO : MISSING ROOT DIR INFORMATION
    


# mark evaluation as finished
torch.save(datetime.datetime.now().time(), OUT_DIR + '/results/' + '/finished_plot.pt')