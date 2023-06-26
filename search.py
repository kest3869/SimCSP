
# Libraries
import os
import sys 
import shutil
import logging
import datetime
import gc
import argparse
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

# Create the argument parser
parser = argparse.ArgumentParser(description='Pretrain model')
parser.add_argument('-b', '--batch_size', type=int, help='The batch size')
parser.add_argument('-l', '--learning_rate', type=float, help='The learning rate')
parser.add_argument('-p', '--model_save_path', type=str, help='The model save path')

# Parse the command line arguments
args = parser.parse_args()

# Retrieve the values of the command line arguments
batch_size = args.batch_size
learning_rate = args.learning_rate
#model_save_path = args.model_save_path

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Paths  
PRETRAIN_DATASET = '/home/pretrain_hrg_validation.hf'
PRETRAINED_MODEL = str()
OUT_DIR = str()

# Converts hyper-parameter to string associated with correct directory 
hp_str = {64:'64', 256:'256', 512:'512', 
          1e-5:'1e5', 3e-5:'3e5', 5e-5:'5e5'}

# Load data first 
# dataset = load_from_disk(PRETRAIN_DATASET)
max_seq_len = 400
use_cl = True
# pretrain_dataset = search_helpers.Prepare_Dataset(dataset, max_seq_len, use_cl)

# Construct OUT_DIR
OUT_DIR = '/home/search_space/hrg_validation/' + hp_str[batch_size] + '/' + hp_str[learning_rate] + '/'

# Pre-training
# PRETRAINED_MODEL = search_helpers.pretrain_model(pretrain_dataset, OUT_DIR, batch_size, learning_rate, max_seq_len)

PRETRAINED_MODEL = OUT_DIR + 'pretrained/'

# Fine-tuning 
search_helpers.finetune_model(PRETRAINED_MODEL, OUT_DIR)

