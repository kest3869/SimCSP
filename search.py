
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

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Paths  
PRETRAIN_DATASET = '/home/pretrain_hrg_train.hf'
PRETRAINED_MODEL = str()
OUT_DIR = str()

# Search_space
bss = [64, 256, 512] # batch sizes 
lrs = [1e-5, 3e-5, 5e-5] # learning rates 

# Converts hyper-parameter to string associated with correct directory 
hp_str = {64:'64', 256:'256', 512:'512', 
          1e-5:'1e5', 3e-5:'3e5', 5e-5:'5e5'}

# Load data first 
dataset = load_from_disk(PRETRAIN_DATASET)
max_seq_len = 400
use_cl = True
pretrain_dataset = search_helpers.Prepare_Dataset(dataset, max_seq_len, use_cl)

# Search loop 
for bs in bss:
    for lr in lrs: 

        # Construct OUT_DIR
        OUT_DIR = '/home/search_space/msg_train/' + hp_str[bs] + '/' + hp_str[lr] + '/'

        # Pre-training
        PRETRAINED_MODEL = search_helpers.pretrain_model(pretrain_dataset, OUT_DIR, bs, lr, max_seq_len)

        # Fine-tuning 
        search_helpers.finetune_model(PRETRAINED_MODEL, OUT_DIR)




