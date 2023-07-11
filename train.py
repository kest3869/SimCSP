
# Mandatory
import os
import sys 
import argparse
import datetime
import argparse
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
import split_spliceator  # splits the dataset for validation during pre-training 
from utils import make_directory # modified files to use python instead of cython and placed in cwd

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

# command line tools
parser = argparse.ArgumentParser(description='Finetune model')
parser.add_argument('-o', '--out-dir', type=str, help='The model save path')
parser.add_argument('-m', '--pretrained-model', type=str, help='Path to pre-trained model')
# Parse the command line arguments
args = parser.parse_args()

# Retrieve the values of the command line argument
OUT_DIR = args.out_dir
PRETRAINED_MODEL = args.pretrained_model

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(OUT_DIR + 'finetune.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# skip if already completed 
if os.path.exists(PRETRAINED_MODEL + 'finished_finetune.pt'):
    logger.info("Found finetuned: Skipping finetune!")
    sys.exit()

# Get device
# Check for CPU or GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# Display device being used for train.py
logger.info(f"Using {device}")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
# Specify the maximum length
max_len = 400
# Load dataset using class from load.py file 
ds = split_spliceator.split_spliceator(False, tokenizer)
labels = split_spliceator.get_labels(ds)

# KFold Splitting
# Hyperparameters (all parts of training)
num_folds = 10 # K in StratifiedKFold
seed = 42 # Random seed for StratifiedKFold
num_train_epochs= 200 # Max number of training epochs for each model 
bs = 16 # batch size used to train the model 
nw = 4 # number of workers used for dataset construction 
resume = False # used to restart model training from a checkpoint
learning_rate = 0.00001 # Learning rate for model training 
wd = 1E-6 # weight decay for model training
patience = 2 # num iterations a model will train without improvements to val_auc 
num_folds_train = 5 # define the number of folds to train (for quicker testing) not in original code 

splits = list()
splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
for _, inds in splitter.split(np.arange(len(ds)), y=labels): # adjusted to work with subset
    splits.append(inds)

best_auc = -1
best_epoch = -1
fold_ckpt = dict()

for epoch in range(num_train_epochs):
    epoch_val_auc = list()
    epoch_val_f1 = list()

    for fold in range(num_folds):

    # CODE ADDED BY AUTHOR Kevin STULL 
        # START ADDED CODE
        used_folds = range(num_folds_train)
        if fold not in used_folds:
            continue
        # END ADDED CODE 

        # setup folder 
        fold_outdir = make_directory(os.path.join(OUT_DIR, "fold{}".format(fold)))
        ckpt = os.path.join(fold_outdir, "checkpoint.pt")
        fold_ckpt[fold] = ckpt

        # setup dataset 
            #(uses a 70/10/20) split for model evaluation
        all_inds = splits[fold:] + splits[:fold]
        train_inds = np.concatenate(all_inds[1:])
        val_inds = all_inds[0]

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

# Model Initialization
        # Not new model 
        if epoch > 0 or (resume and os.path.exists(ckpt)):
            if epoch > 0:
                del model, optimizer, scaler
            d = torch.load(ckpt)
            model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=1).to(device)
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
            model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=1).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=wd
            )
            torch.save((train_inds, val_inds), "{}/split.pt".format(OUT_DIR))
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


################# 

        # Added by Author (Kevin Stull) For Convinience
        torch.save((epoch_val_f1, epoch_val_auc), os.path.join(fold_outdir, "val_metrics.pt"))

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
        logger.info("NEW BEST: " + str(best_auc) + ' at epoch ' + str(best_epoch))
        for fold in range(10):
            if fold not in used_folds: # added this since I may not use all the folds 
                continue
            ckpt = fold_ckpt[fold]
            shutil.copy2(ckpt, "{}.best_model.pt".format(ckpt))
            wait = 0
    else:
        wait += 1
        if wait >= patience:
            break


# Save hyperparameter info to the logger
metadata = {
    'learning_rate': learning_rate,
    'batch_size': bs,
    'patience': patience,
    'optimizer': 'AdamW',
    'loss':'Accuracy',
    'len(ds)': len(ds),
    'outdir':OUT_DIR,
    'pretrained_model_path':PRETRAINED_MODEL,
    'number examples:':len(ds),
    'time':datetime.datetime.now().time()
    }

# Mark training as finished
torch.save(datetime.datetime.now().time(), OUT_DIR + 'finished_finetune.pt')
logger.info('Finished with hyperparameters: %s', metadata)

# END OF CITATION