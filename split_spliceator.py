# libraries
import torch
import numpy as np
from tqdm import tqdm
import random
from sentence_transformers import InputExample
from sklearn.model_selection import StratifiedKFold, train_test_split

# files
import load

# loads, splits, and returns three list of indices (fold_num, 1)
def split_spliceator(labels, OUT_DIR, num_folds=5, rng_seed=42):
    '''
    Input: 
    - OUT_DIR : directory where splits are saved 
    - labels : list containing labels of dataset    
    - num_folds : the number of folds to make (DEFAULT=5)
    - rng_seed : allows for deterministic splitting of dataset (DEFAULT=42)
    Output: 
    - train_split, validation_split, test-split : (num_folds, 1) lists each containing indices from original ds 
    '''

    # create a k-fold cross validator 
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=rng_seed)
    # generate folds for dataset
        # default behavior: (80/20) : (train/test)
    folds = skf.split(np.zeros(len(labels)), labels) 
    # list to save the folds
    temp_ind = []
    # get the folds 
    for fold in folds:
        temp_ind.append(fold)
    # holds our folds 
    train_split = list()
    validation_split = list()
    test_split = list()
    for fold_num in range(num_folds):
        # split dataset (train/validation/test):(70/10/20)
        train_ind, validation_ind = train_test_split(temp_ind[fold_num][0], train_size=0.875, random_state=rng_seed)
        # get test split
        test_ind = temp_ind[fold_num][1]
        # save the splits to a list 
        train_split.append(train_ind)
        validation_split.append(validation_ind)
        test_split.append(test_ind)

    # save the folds for training 
    torch.save(train_split, OUT_DIR + '/train_split.pt')
    torch.save(validation_split, OUT_DIR + '/validation_split.pt')
    torch.save(test_split, OUT_DIR + '/test_split.pt')

    return train_split, validation_split, test_split

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
    random_pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices), 2)]

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

if __name__ == "__main__":

    # libraries 
    import os
    import argparse
    import logging
    import numpy as np
    import datetime
    import torch 
    from transformers import AutoTokenizer

    # files
    import load 

    # command line tools
    parser = argparse.ArgumentParser(description='Generate splits for Spliceator dataset')
    parser.add_argument('--split_dir', type=str, help='The path to the splits')
    # Parse the command line arguments
    args = parser.parse_args()
    # Retrieve the values of the command line argument
    SPLIT_DIR = args.split_dir
    
    # create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(SPLIT_DIR + 'split.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # arguments 
    num_folds = 5 # K in StratifiedKFold
    seed = np.random.randint(1, 1E5) # random value used to generate splits
    
    # load dataset
    tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')
    positive_dir = '/storage/store/kevin/data/spliceator/Training_data/Positive/GS'
    negative_dir = '/storage/store/kevin/data/spliceator/Training_data/Negative/GS/GS_1'
    positive_files = [os.path.join(positive_dir, file) for file in os.listdir(positive_dir)]
    negative_files = [os.path.join(negative_dir, file) for file in os.listdir(negative_dir)]
    ds = load.SpliceatorDataset(
        positive=positive_files,
        negative=negative_files,
        tokenizer=tokenizer,
        max_len=400
    )

    # call split_spliceator to save splits  
    split_spliceator(ds.labels, SPLIT_DIR, num_folds, seed)

    # save num_folds, seed, time
    logger.info("num_folds: " + str(num_folds))
    logger.info("seed: " + str(seed))
    logger.info("time: " + str(datetime.datetime.now().time()))
    logger.info("split_dir: " + SPLIT_DIR)