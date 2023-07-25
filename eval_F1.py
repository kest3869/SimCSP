# libraries
import argparse
import logging
import os
import sys 
import csv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import datetime
import torch
import torch.nn
from torch.utils.data import Subset, DataLoader
from torch.cuda.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# files
from get_paths import get_paths
import load

# Test Function
@torch.no_grad()
@autocast()
def test_model(model: AutoModelForSequenceClassification, loader: DataLoader):
    """
    Return:
    auc : float
    f1 : float
    pred : list
    true : list
    """
    model.eval()
    pred, true = list(), list()
    for it, (ids, mask, label) in enumerate(tqdm(loader, desc="predicting", total=len(loader))):
        ids = ids.to(device)
        mask = mask.to(device)
        score = torch.sigmoid(model.forward(ids, attention_mask=mask).logits.squeeze(1)).detach().cpu().numpy()
        del ids
        label = label.numpy()
        pred.append(score.astype(np.float16))
        true.append(label.astype(np.float16))
    pred = np.concatenate(pred)
    true = np.concatenate(true)
    auc_list = roc_auc_score(true.T, pred.T)
    f1 = f1_score(true.T, pred.T > 0.5)
    return auc_list, f1, pred, true

# return a list of F1 scores given a list of models and a dataset
def get_F1_scores(model_paths, ds_eval):
    """
    Input:
    - model_paths : a list of models for sequence classification
    - ds_eval : a labelled torch dataset or subset used for evaluation of models 
    Outputs:
    - f1_scores : a list of F1 scores as a np.array
    - model_paths : a list of paths used to generate the models as a np.array
    """
    f1_scores = []
    # get F1 scores for all models in model paths 
    for model_path in model_paths:
        # initialize the dataloader 
        val_loader = DataLoader(
            ds_eval,
            batch_size=8,
            num_workers=4
        )
        # load pre-trained model 
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1).to(device)
        # get f1_score 
        _, f1, _, _ = test_model(model, val_loader)
        # save f1 score
        f1_scores.append(f1)
    # convert lists to np.arrays
    return f1_scores, model_paths 

# Create the argument parser
parser = argparse.ArgumentParser(description='Get_F1_scores')
parser.add_argument('-p', '--out_dir', type=str, help='The save path of the F1 scores')
parser.add_argument('--split_dir', type=str, help='The path to the splits')
# Parse the command line arguments
args = parser.parse_args()
# Retrieve the values of the command line arguments
OUT_DIR = args.out_dir
SPLIT_DIR = args.split_dir


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(OUT_DIR + '/results/' + 'eval_F1.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# skip if already completed 
if os.path.exists(OUT_DIR + '/results/' + '/finished_F1.pt'):
    logger.info("Found finished, skipping F1.")
    print("Found finished, skipping eval_F1.py!")
    sys.exit()

# Get device
# Check for CPU or GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# get paths to trained models 
_, finetuned_models = get_paths(OUT_DIR)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/storage/store/kevin/data/tokenizer_setup")
# Positive and Negative paths
positive_dir = '/storage/store/kevin/data/spliceator/Training_data/Positive/GS'
negative_dir = '/storage/store/kevin/data/spliceator/Training_data/Negative/GS/GS_1'
# List all files in the directory
positive_files = [os.path.join(positive_dir, file) for file in os.listdir(positive_dir)]
negative_files = [os.path.join(negative_dir, file) for file in os.listdir(negative_dir)]
# Load dataset using class from load.py file
ds = load.SpliceatorDataset(
    positive=positive_files,
    negative=negative_files,
    tokenizer=tokenizer,
    max_len=400
)

# load the split the fine-tuned model used 
test_split = torch.load(SPLIT_DIR + 'test_split.pt')

# get the F1 scores for each fold 
f1s = []
paths = []
mean_f1s = []
# loop through each fold
for i in range(np.shape(test_split)[0]):
    # get correct subset of ds for fold i
    test_ds = Subset(ds, test_split[i])
    # calculate F1 scores
    temp_f1s, temp_paths = get_F1_scores(finetuned_models, test_ds)
    paths.append(temp_paths)
    f1s.append(temp_f1s)

# save CSV of F1 scores 
with open(OUT_DIR + '/results/' + 'F1_results_.csv', "w", newline="") as file1:
    writer = csv.writer(file1)
    writer.writerows(f1s)
# save CSV of paths
with open(OUT_DIR + '/results/' + 'F1_paths_.csv', "w", newline="") as file3:
    writer = csv.writer(file3)
    writer.writerows(paths)

# mark evaluation as finished
torch.save(datetime.datetime.now().time(), OUT_DIR + '/results/' + '/finished_F1.pt')
