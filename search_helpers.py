# Libraries
import os
import sys 
import shutil
import logging
import datetime
import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, Subset 
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F 
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, BertTokenizer
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, losses, models, InputExample
from tqdm import tqdm

# Files 
import load # this is my (Kevin Stull) version of the SpliceBERT code for loading Spliceator data
from utils import make_directory # modified files to use python instead of cython and placed in cwd
import search_helpers


class Prepare_Dataset(Dataset):
    def __init__(self, original_dataset, max_seq_len, use_contrastive_learning=False):
        self.original_dataset = original_dataset
        self.max_seq_len = max_seq_len
        self.use_contrastive_learning = use_contrastive_learning
        self.data = self.generate_new_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def generate_new_dataset(self):
        new_dataset = []
        for a in self.original_dataset:
            start = np.random.randint(0, 100) # following convention of TNT
            for i in range(5900 // self.max_seq_len): # 6000 is max length of seq. in dataset 
                element = a['sequence'][start + i * self.max_seq_len: start + (i + 1) * self.max_seq_len]
                if 'N' not in element: # Throw out exmples containing N
                    if self.use_contrastive_learning: # saves as a tuple for contrastive learning 
                        new_dataset.append(InputExample(texts=[element, element]))
                    else: # save as a single element for MLM
                        new_dataset.append(InputExample(texts=[element]))
        return new_dataset
    

def pretrain_model(pretrain_dataset, OUT_DIR, bs, lr, max_seq_len):
    '''
    Inputs: 
    - PRETRAIN_DATASET : path to data used for pre-training
    - OUT_DIR : path to location where model is saved
    - batch_size : size of training batch used for Contastive Learning 
    - learning_rate : learning rate during pre-training
    Output:
    - PRETRAINED_MODEL : the path to the pre-trained model  
    '''

    # skip if already completed 
    PRETRAINED_MODEL = OUT_DIR + 'pretrained/'
    if os.path.exists(PRETRAINED_MODEL + 'finished.pt'):
        return PRETRAINED_MODEL
    # make the directory if it does not exist 
    if not os.path.exists(PRETRAINED_MODEL):
        os.makedirs(PRETRAINED_MODEL)

    # load data 
    '''
    dataset = load_from_disk(PRETRAIN_DATASET)
    use_cl = True
    pretrain_ds = Prepare_Dataset(dataset, max_seq_len, use_cl)
    '''
    data_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=bs, shuffle=True)
    
    # define model 
    model_path = "/home/SpliceBERT.510nt/"  
    word_embedding_model = models.Transformer(model_path, max_seq_length=max_seq_len)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # number of training epochs 
    num_epochs = 1
 
    # learning rate 
    optimizer_class = AdamW
    optimizer_params =  {'lr': lr}

    # fit model
    model.fit(
        train_objectives=[(data_loader, train_loss)],
        epochs=num_epochs,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        output_path=PRETRAINED_MODEL
    )

    # mark training as finished
    torch.save(datetime.datetime.now().time(), OUT_DIR + 'finished.pt')

    return PRETRAINED_MODEL


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
    # Set value for device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # not gradient computations needed
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


def finetune_model(PRETRAINED_MODEL, OUT_DIR):
    """
    Input:
    - PRETRAINED_MODEL: path to a pretrained SpliceBERT style model
    - OUT_DIR: path to directory where models will be saved 
    OUTPUT:
    - FINETUNED_MODELS : a path to a directory of fine-tuned models
    """

# Checks before training
    # output directory 
    FINETUNED_MODEL = OUT_DIR + '/finetuned/' 
    # skip if already completed 
    if os.path.exists(FINETUNED_MODEL + 'finished.pt'):
        return FINETUNED_MODEL
    # if OUT_DIR does not exist at all, create it
    if not os.path.exists(FINETUNED_MODEL):
        os.makedirs(FINETUNED_MODEL)
    # Set the logging level to suppress output
    logging.getLogger("transformers").setLevel(logging.ERROR)

# Get Device
    # set value for device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

# Load Dataset 
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    # Specify the directory path
    positive_dir = '/home/spliceator/Training_data/Positive/GS'
    negative_dir = '/home/spliceator/Training_data/Negative/GS/GS_1'
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
    bs = 16 # batch size used to train the model 
    nw = 4 # number of workers used for dataset construction 
    resume = True # used to restart model training from a checkpoint
    learning_rate = 0.00001 # Learning rate for model training 
    wd = 1E-6 # weight decay for model training
    patience = 1 # num iterations a model will train without improvements to val_auc 
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
            # setup folder 
            fold_outdir = make_directory(os.path.join(FINETUNED_MODEL, "fold{}".format(fold)))
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
                torch.save((train_inds, val_inds, test_inds), "{}/split.pt".format(FINETUNED_MODEL))
                scaler = GradScaler()
                trained_epochs = 0

            # Training
            model.train()
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
                ckpt = fold_ckpt[fold]
                shutil.copy2(ckpt, "{}.best_model.pt".format(ckpt))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

# END OF CITATION
    # mark training as finished
    torch.save(datetime.datetime.now().time(), FINETUNED_MODEL + 'finished1.pt')