
'''
START OF CITATION

CODE ADAPTED FROM: 
https://github.com/biomed-AI/SpliceBERT/blob/main/examples/04-splicesite-prediction/spliceator_data.py

ORIGINAL MANUSCRIPT: 
@article{Chen2023.01.31.526427,
	author = {Chen, Ken and Zhou, Yue and Ding, Maolin and Wang, Yu and Ren, Zhixiang and Yang, Yuedong},
	title = {Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction},
	year = {2023},
	doi = {10.1101/2023.01.31.526427},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/02/03/2023.01.31.526427},
	journal = {bioRxiv}
}

ADAPTED BY Kevin Stull
'''

# Mandatory 
import os
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Files 
from utils import make_directory
import load # using a load function adapted by Kevin Stull 

# Optional 
import sys
import pickle
import argparse
import pandas as pd
import torch.nn.functional as F
from utils import make_logger, get_run_info

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
    for it, (ids, mask, label) in enumerate(tqdm(loader, desc="Predicting", total=len(loader))):
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



if __name__ == "__main__":

    OUT_DIR = '/storage/store/kevin/local_files/exp5/BEST_SCCS/'

    # make directory if it does not exist 
    if not os.path.exists(OUT_DIR + '/results/inference/'):
        os.makedirs(OUT_DIR + '/results/inference/')
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(OUT_DIR + '/results/inference/' + 'benchmarking.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test each species 
    species_list = ['Danio', 'Fly', 'Thaliana', 'Worm']
    for species in species_list:

        # Filepath to model initialization parameters 
        SPLICEBERT_PATH = "/storage/store/kevin/data/SpliceBERT-human.510nt"
        # Specify the directory path
        p_a = "/storage/store/kevin/data/spliceator/Benchmarks/" + species + "/SA_sequences_acceptor_400_Final_3.positive.txt"
        p_d = "/storage/store/kevin/data/spliceator/Benchmarks/" + species + "/SA_sequences_donor_400_Final_3.positive.txt"
        n_a = "/storage/store/kevin/data/spliceator/Benchmarks/" + species + "/SA_sequences_acceptor_400_Final_3.negative.txt"
        n_d = "/storage/store/kevin/data/spliceator/Benchmarks/" + species + "/SA_sequences_donor_400_Final_3.negative.txt"
        positive_files = [p_a, p_d]
        negative_files = [n_a, n_d]

        # get the scores for each fold 
        auc_scores, ap_scores, f1s = [], [], []
        for fold in range(5):

            # Path to best model (currently the best version of only fold0 model) 
            BEST_MODEL_PATH = "/storage/store/kevin/local_files/exp5/BEST_SCCS/finetuned_models/fold" + str(fold) + '/checkpoint.pt.best_model.pt' 

            # initialize tokenizer 
            tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)

            # Hyperparameters
            ml = 400 # Maximum input length 
            bs = 16 # batch size used to train the model 
            nw = 4 # number of workers used for dataset construction 

            # load dataset
            ds = load.SpliceatorDataset(
                positive=positive_files,
                negative=negative_files,
                tokenizer= tokenizer,
                max_len=ml
            )

            train_loader = DataLoader(
                ds,
                batch_size = bs,
                num_workers = nw
            )

            model = AutoModelForSequenceClassification.from_pretrained(SPLICEBERT_PATH, num_labels=1).to(device)
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu")["model"]) 
            model.eval()

            # Evaluation
            pbar = tqdm(train_loader, total=len(train_loader))
            with torch.no_grad():
                all_scores, all_labels = list(), list()
                for it, (ids, mask, label) in enumerate(pbar):
                    ids, mask = ids.to(device), mask.to(device)
                    with autocast():
                        logits = model.forward(ids, attention_mask=mask).logits.squeeze(1)
                        all_scores.append(logits.detach().cpu().numpy())
                        all_labels.append(label.detach().cpu().numpy())
            
            all_scores = np.concatenate(all_scores)
            all_labels = np.concatenate(all_labels)
            pickle.dump((all_labels, all_scores), open("{}/results.pkl".format(OUT_DIR + '/results/inference/'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            auc_score = roc_auc_score(all_labels, all_scores)
            ap_score = average_precision_score(all_labels, all_scores)
            f1 = f1_score(all_labels, all_scores > 0.5)

            logger.info("AUC/AUPR/F1: {:.4f} {:.4f} {:.4f}".format(auc_score, ap_score, f1))

            auc_scores.append(auc_score)
            ap_scores.append(ap_score)
            f1s.append(f1)

        logger.info("model: %s", BEST_MODEL_PATH)
        logger.info("species: %s", species)
        logger.info("mean auc_score: %.4f", np.mean(auc_scores))
        logger.info("mean ap_score: %.4f", np.mean(ap_scores))
        logger.info("mean f1: %.4f", np.mean(f1s))

