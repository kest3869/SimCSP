
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
Code is derivative of (Cited Author) unless otherwise stated. 
'''

# Mandatory
import os
import sys 
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import torch 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Subset 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
import logging

# Files 
import load # this is my (Kevin Stull) version of the SpliceBERT code for loading Spliceator data
from utils import make_directory # modified files to use python instead of cython and placed in cwd

# Optional 
import pandas as pd
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoConfig

# Set the logging level to suppress output
logging.getLogger("transformers").setLevel(logging.ERROR)

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

if __name__ == "__main__":

# Get device
    # Check for CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Display device being used for train.py
    print("Using", device)

# Load Dataset 
    # Set the path to the folder of pre-trained SpliceBERT
    SPLICEBERT_PATH = os.getcwd() + "/exp1"
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)
    # Specify the directory path
    positive_dir = os.getcwd() + '/spliceator/Training_data/Positive/GS'
    negative_dir = os.getcwd() + '/spliceator/Training_data/Negative/GS/GS_1'
    # List all files in the directory
    positive_files = [os.path.join(positive_dir, file) for file in os.listdir(positive_dir)]
    negative_files = [os.path.join(negative_dir, file) for file in os.listdir(negative_dir)]
    # Specify the maximum length
    max_len = 400
    # Load dataset using class from load.py file 
    ds = load.SpliceatorDataset(
        positive=positive_files, 
        negative=negative_files, 
        tokenizer=tokenizer, 
        max_len=max_len
    )

# KFold Splitting

    # Hyperparameters (all parts of training)
    num_folds = 10 # K in StratifiedKFold
    seed = 42 # Random seed for StratifiedKFold
    num_train_epochs= 200 # Max number of training epochs for each model 
    out_dir = os.getcwd() + '/results_experimental/' # output directory for results of train.py
    bs = 16 # batch size used to train the model 
    nw = 4 # number of workers used for dataset construction 
    resume = True # used to restart model training from a checkpoint
    learning_rate = 0.00001 # Learning rate for model training 
    wd = 1E-6 # weight decay for model training
    patience = 5 # num iterations a model will train without improvements to val_auc 

    splits = list()
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for _, inds in splitter.split(np.arange(len(ds)), y=ds.labels):
        splits.append(inds)
    
    best_auc = -1
    best_epoch = -1
    fold_ckpt = dict()

    for epoch in range(num_train_epochs):
        epoch_val_auc = list()
        epoch_val_f1 = list()
        epoch_test_auc = list()
        epoch_test_f1 = list() 

        for fold in range(num_folds):

            # Added by author for quick model training 
            if not fold == 0:
                continue 

            # setup folder 
            fold_outdir = make_directory(os.path.join(out_dir, "fold{}".format(fold)))
            ckpt = os.path.join(fold_outdir, "checkpoint.pt")
            fold_ckpt[fold] = ckpt

            # setup dataset 
                #(uses a 70/10/20) split for model evaluation
            all_inds = splits[fold:] + splits[:fold]
            train_inds = np.concatenate(all_inds[3:])
            val_inds = all_inds[0]
            test_inds = np.concatenate(all_inds[1:3])

            # Loading datasets
            train_loader = DataLoader(
                Subset(ds, indices=train_inds),
                batch_size=bs,
                shuffle=True,
                drop_last=True,
                num_workers=nw,
            )
            val_loader = DataLoader(
                Subset(ds, indices=val_inds),
                batch_size=bs,
                num_workers=nw
            )
            test_loader = DataLoader(
                Subset(ds, indices=test_inds),
                batch_size=bs,
                num_workers=nw
            )

# Model Initialization
            # Not new model 
            if epoch > 0 or (resume and os.path.exists(ckpt)):
                if epoch > 0:
                    del model, optimizer, scaler
                d = torch.load(ckpt)
                model = AutoModelForSequenceClassification.from_pretrained(SPLICEBERT_PATH, num_labels=1).to(device)
                model.load_state_dict(d["model"])
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=wd
                )
                optimizer.load_state_dict(d["optimizer"])
                scaler = GradScaler()
                scaler.load_state_dict(d["scaler"])
                if epoch == 0:
                    trained_epochs = d.get("epoch", -1) + 1
            # New model 
            else: 
                model = AutoModelForSequenceClassification.from_pretrained(SPLICEBERT_PATH, num_labels=1).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=wd
                )
                torch.save((train_inds, val_inds, test_inds), "{}/split.pt".format(out_dir))
                scaler = GradScaler()
                trained_epochs = 0

            model.train()

            # Training
            pbar = tqdm(
                train_loader,
                total=len(train_loader),
                desc="Epoch{}-{}".format(epoch+trained_epochs, fold)
                        )
            epoch_loss = 0

            for it, (ids, mask, label) in enumerate(pbar):
                ids, mask, label = ids.to(device), mask.to(device), label.to(device).float()
                optimizer.zero_grad()
                with autocast():
                    logits = model.forward(ids, attention_mask=mask).logits.squeeze(1)
                    loss = F.binary_cross_entropy_with_logits(logits, label).mean()
                
                scaler.scale(loss).backward() 
                scaler.step(optimizer) 
                scaler.update() 

                epoch_loss += loss.item()

                # Loss per batch 
                pbar.set_postfix_str("loss/lr={:.4f}/{:.2e}".format(
                    epoch_loss / (it + 1), optimizer.param_groups[-1]["lr"]
                ))

            # Validation
            val_auc, val_f1, val_score, val_label = test_model(model, val_loader)
            torch.save((val_score, val_label), os.path.join(fold_outdir, "val.pt"))
            epoch_val_auc.append(val_auc)
            epoch_val_f1.append(val_f1)

            # Testing
            test_auc, test_f1, test_score, test_label = test_model(model, test_loader)
            torch.save((test_score, test_label), os.path.join(fold_outdir, "test.pt"))
            epoch_test_auc.append(test_auc)
            epoch_test_f1.append(test_f1)



################# 

            # Added by Author (Kevin Stull) For Convinience
                # NOT IN ORIGINAL CODE! 
            torch.save((val_f1, val_auc), os.path.join(fold_outdir, "val_metrics.pt"))
            torch.save((test_f1, test_auc), os.path.join(fold_outdir, "test_metrics.pt"))

################# 



            # Save model
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch
            }, ckpt)

# Checking for new best model 
        if np.mean(epoch_val_auc) > best_auc:
            best_auc = np.mean(epoch_val_auc)
            best_epoch = epoch
            for fold in range(10):
                # added for quick model training 
                if not fold == 0:
                    continue 
                ckpt = fold_ckpt[fold]
                shutil.copy2(ckpt, "{}.best_model.pt".format(ckpt))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

# END OF CITATION