# libraries
import os
from torch.utils.data import random_split
import torch

# files
import load

# loads, splits, and returns a subset of a spliceator dataset
def split_spliceator(for_pretrain, tokenizer, rng_seed=42):

    '''
    Input: 
    - for_pretrain : boolean value that determines if pre-train or train split is returned 
    - toknizer : torch tokenizer used to build the dataset object
    - rng_seed : allows for deterministic splitting of dataset
    Output: 
    - a spliceator dataset half the size of the original 
    '''

    # Load Dataset
    # Positive and Negative paths
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
    # Fixes the rng
    generator = torch.Generator().manual_seed(rng_seed)

    # calculating the lengths 
    ds_len = len(ds)
    sub_len = ds_len // 2

    # splits the dataset
    datasets = random_split(ds, [sub_len, sub_len], generator=generator)

    return datasets[0] if for_pretrain else datasets[1]

