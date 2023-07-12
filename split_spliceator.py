# libraries
import os
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
from tqdm import tqdm
import random
from sentence_transformers import InputExample

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
    positive_dir = '/home/data/spliceator/Training_data/Positive/GS'
    negative_dir = '/home/data/spliceator/Training_data/Negative/GS/GS_1'
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
    pretrain_ds, train_ds = random_split(ds, [sub_len, sub_len], generator=generator)

    # process data differently based on destination 
    if for_pretrain:
        return pretrain_ds 
    else:
        return train_ds

# prepares sentence-transformers compatible version of spliceator dataset
def prep_val_data(ds, tokenizer, rng_seed=42):
    '''
    Input: 
    - ds : a spliceator dataset (or subset)
    - tokenizer : torch tokenizer used to encode sequences 
    - rng_seed : seed used to create the pairs of data points 
    Output:
    - new_dataset : list of Sentence-Transformer InputExample objects 
    '''

    # generate the pairs of indices
    n = len(ds)  # length of dataset
    indices = list(range(n))  # indices of dataset
    random.seed(rng_seed)  # Set the random seed to 42 for reproducibility
    random.shuffle(indices)  # shuffle the indices randomly
    if len(indices) % 2 != 0:  # If the length is odd
        indices.pop(0)  # Remove the first element
    # generate tuples from the shuffled list 
    random_pairs = tqdm([(indices[i], indices[i+1]) for i in range(0, len(indices), 2)], desc='making pairs')

    # new "dataset" 
    new_dataset = []

    # build new dataset
    for pair in tqdm(random_pairs, desc='building new dataset'):
        # get seq and label associated with index from ds
        seq1, _, label1 = ds[pair[0]]
        seq2, _, label2 = ds[pair[1]]
        # Sentence Transformers expects an un-tokenized input 
        seq1 = tokenizer.decode(seq1)
        seq2 = tokenizer.decode(seq2)
        # if they have the same label, then they should be considered "similar"
        if label1 == label2:
            label = 1 # similar
        else:
            label = 0 # NOT similar 
        # sentence tranformer compatible validation example 
        new_datapoint = InputExample(texts=[seq1, seq2], label=label)
        # add to the new dataset
        new_dataset.append(new_datapoint)
    # express as numpy array 
    new_dataset = np.array(new_dataset, dtype=object)
    
    return new_dataset

# takes a torch subset, returns a list of its associated labels 
def get_labels(subset):

    labels = []
    # get the labels 
    for index in tqdm(subset.indices, desc='getting labels'):
        _, _, label = subset.dataset[index]
        labels.append(label)
    # convert to numpy array 
    labels = np.array(labels)
    return labels 