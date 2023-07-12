
# practice using StratifiedKFold, Subset, and train_test_split

# libraries
import os
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset 
# files
import load


# Load Dataset
# Positive and Negative paths
positive_dir = '/home/data/spliceator/Training_data/Positive/GS'
negative_dir = '/home/data/spliceator/Training_data/Negative/GS/GS_1'
# List all files in the directory
positive_files = [os.path.join(positive_dir, file) for file in os.listdir(positive_dir)]
negative_files = [os.path.join(negative_dir, file) for file in os.listdir(negative_dir)]
# Specify the maximum length
max_len = 400
tokenizer = AutoTokenizer.from_pretrained('/home/data/tokenizer_setup')
# Load dataset using class from load.py file
ds = load.SpliceatorDataset(
    positive=positive_files,
    negative=negative_files,
    tokenizer=tokenizer,
    max_len=max_len
)

# create a 5-fold cross validator 
skf = StratifiedKFold(n_splits=5, shuffle=True)
# generate folds for dataset
folds = skf.split(ds.sequences, ds.labels)
# list to save the folds
fold_ind = []
# print the folds 
for fold in folds:
    print(fold)
    fold_ind.append(fold)
# save the folds 
torch.save(fold_ind, '/home/local_files/scrap/folds.pt')
print('save and load')
# load the folds
test = torch.load('/home/local_files/scrap/folds.pt')
# print the loaded folds
for fold in test:
    print(fold[False], fold[True])
