
# Mandatory
import os
import sys 
import shutil
import argparse
import logging
import datetime
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import torch 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Subset 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler

# Files 
import load # loads the spliceator dataset 
from split_spliceator import split_spliceator  # splits the dataset and saves the indices 

# Environment 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# used to add tokenizer meta-data to folds for embed_gen.py
def copy_files(source_folder, destination_folder):
    # Get the list of files in the source folder
    file_list = os.listdir(source_folder)
    # Copy each file from the source folder to the destination folder
    for file_name in file_list:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy2(source_path, destination_path)

# Test Function
@torch.no_grad()
@autocast()
def validate_model(model: AutoModelForSequenceClassification, loader: DataLoader):
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
OUT_DIR += '/finetuned_models/'
PRETRAINED_MODEL = args.pretrained_model
PRETRAINED_MODEL += '/pretrained_models/'

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(OUT_DIR + 'finetune.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# skip if already completed 
if os.path.exists(OUT_DIR + 'finished_finetune.pt'):
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

# Hyperparameters (all parts of training)
num_folds = 2 # K in StratifiedKFold
seed = np.random.randint(0,1E5) # Random seed for StratifiedKFold and train_test_split
num_train_epochs = 5 # Max number of training epochs for each model 
bs = 64 # batch size used to train the model [Default 16]
nw = 4 # number of workers used for dataset construction 
resume = False # used to restart model training from a checkpoint
learning_rate = 1E-5 # Learning rate for model training 
wd = 1E-6 # weight decay for model training
patience = 2 # num iterations a model will train without improvements to val_auc 
max_len = 400

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

# Load dataset
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
    max_len=max_len
)
# call split_spliceator to get our splits 
train_split, validation_split, test_split = split_spliceator(ds.labels, OUT_DIR, num_folds=num_folds, rng_seed=seed)

# used for model evaluation
best_auc = -1
best_epoch = -1
fold_ckpt = dict()

for epoch in range(num_train_epochs):
    epoch_val_auc = list()
    epoch_val_f1 = list()

    for fold in range(num_folds):

        # make paths 
        fold_outdir = os.path.join(OUT_DIR, "fold{}".format(fold))

        # setup folder 
        if not os.path.exists(fold_outdir):
            os.makedirs(fold_outdir)
            # add information about tokenizer for embed_gen.py
            copy_files('/storage/store/kevin/data/tokenizer_setup', os.path.join(fold_outdir))  

        # update checkpoint
        ckpt = os.path.join(fold_outdir, "checkpoint.pt")
        fold_ckpt[fold] = ckpt

        # Load datasets
        train_loader = DataLoader(
            Subset(ds, indices=train_split[fold]),
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            num_workers=nw,
        )
        val_loader = DataLoader(
            Subset(ds, indices=validation_split[fold]),
            batch_size=bs,
            num_workers=nw
        )

# Model initialization/loading
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
            scaler = GradScaler()
            trained_epochs = 0
        # put model in training mode
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
        val_auc, val_f1, val_score, val_label = validate_model(model, val_loader)
        torch.save((val_score, val_label), os.path.join(fold_outdir, "val.pt"))
        epoch_val_auc.append(val_auc)
        epoch_val_f1.append(val_f1)
        torch.save((epoch_val_f1, epoch_val_auc), os.path.join(fold_outdir, "val_metrics.pt"))

        # making a copy of model for evaluation by epoch later
        epoch_outdir = os.path.join(fold_outdir, "epoch{}".format(epoch))
        # setup folder 
        if not os.path.exists(epoch_outdir):
            os.makedirs(epoch_outdir)
            # add information about tokenizer for embed_gen.py
            copy_files('/storage/store/kevin/data/tokenizer_setup', os.path.join(epoch_outdir)) 
        # make a 2nd checkpoint based on epoch 
        epoch_ckpt = os.path.join(epoch_outdir, "epoch_checkpoint.pt")
        # save the checkpoint for metric tracking later 
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch
        }, epoch_ckpt)
        # additional info for gen_embed.py
        torch.save(model.state_dict(), os.path.join(epoch_outdir, "pytorch_model.bin"))

        # Save model for next round of training
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
        for fold in range(num_folds):
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
