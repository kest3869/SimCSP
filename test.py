
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
import numpy as np
import pickle
import logging
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Files 
from utils import make_directory
import load # using a load function adapted by Kevin Stull 

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

    # get device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    # Filepath to model initialization parameters 
    species = ['Danio', 'Fly', 'Thaliana', 'Worm']
    folds = range(5)

    # initialize tokenizer 
    tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')

    # Hyperparameters
    ml = 400 # Maximum input length 
    bs = 16 # batch size used to train the model 
    nw = 4 # number of workers used for dataset construction 

    # Path to best model
    BEST_MODEL_PATH_TEMPLATE = '/storage/store/kevin/local_files/exp5/BASELINE/finetuned_models/fold{}/checkpoint.pt.best_model.pt' 
    # Inference Out Directory
    INFERENCE_OUT_PATH = "/storage/store/kevin/local_files/exp5/inference/"

    # Check if the directory exists
    if not os.path.exists(INFERENCE_OUT_PATH):
        # If not, create the directory
        os.makedirs(INFERENCE_OUT_PATH)

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(INFERENCE_OUT_PATH + 'inference.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a new CSV file and write headers
    with open('f1_benchmarks.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Species", "Average F1"])

    for species_name in species:
        f1s=[]
        for fold in folds:
            # Specify the directory path
            pa = '/storage/store/kevin/data/spliceator/Benchmarks/' + species_name + '/SA_sequences_acceptor_400_Final_3.positive.txt'
            pd = '/storage/store/kevin/data/spliceator/Benchmarks/' + species_name + '/SA_sequences_donor_400_Final_3.positive.txt'
            na = '/storage/store/kevin/data/spliceator/Benchmarks/' + species_name + '/SA_sequences_acceptor_400_Final_3.negative.txt'
            nd = '/storage/store/kevin/data/spliceator/Benchmarks/' + species_name + '/SA_sequences_donor_400_Final_3.negative.txt'


            # load dataset
            ds = load.SpliceatorDataset(
                positive=[pa, pd],
                negative=[na, nd],
                tokenizer= tokenizer,
                max_len=ml
            )

            bench_loader = DataLoader(
                ds,
                batch_size = bs,
                num_workers = nw
            )

            BEST_MODEL_PATH = BEST_MODEL_PATH_TEMPLATE.format(fold)
            model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_PATH, num_labels=1).to(device)
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu")["model"]) 
            model.eval()

            # Evaluation
            pbar = tqdm(bench_loader, total=len(bench_loader))
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
            pickle.dump((all_labels, all_scores), open("{}/results_{}_fold{}.pkl".format(INFERENCE_OUT_PATH, species_name, fold), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            auc_score = roc_auc_score(all_labels, all_scores)
            ap_score = average_precision_score(all_labels, all_scores)
            f1 = f1_score(all_labels, all_scores > 0.5)
            f1s.append(f1)

            print(f"Species: {species_name}, Fold: {fold}, AUC/AUPR/F1: {auc_score:.4f} {ap_score:.4f} {f1:.4f}")
        f1_final = np.mean(f1s)
        print("F1_final for species", species_name, f1_final)
        logger.info("F1_final for species" + species_name + str(f1_final))
        # Write the species name and average f1 score to the CSV file
        with open('f1_benchmarks.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([species_name, f1_final])

    metadata = {
        'pretrained_model': BEST_MODEL_PATH_TEMPLATE, 
        'max_input_length': ml,  
        'batch_size': bs, 
    }

    logger.info(metadata)