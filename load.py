
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
import numpy as np
import os
import torch
import torch.nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

# Optional 
import pickle
import sys
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizerFast

# Loads the Spliceator into a torch dataset object 
class SpliceatorDataset(Dataset):

    # Comments by Kevin Stull 
    # max length input to accept for data (400 for spliceator)
    # positive : path to positive data (acceptor and donor CSV)
    # negative : path to negative data (acceptor and donor CSV) 
    # tokenizer : a tokenizer initialized from the transformers AutoTokenizer library
    # group : species of origin
    # label : 1 contains splice site, 0 does not contain splice site 
    # sequences : list of DNA sequences (accociated by order with label)

    # process: given filepaths, stores data in class 
    # __getitem__ : given index returns input_id, mask, label 
    # __len__: returns number of training examples in dataset 
    # collate_fn : takes input of [ids, mask, label] returns 

    # input_ids : the tokenized version of the DNA sequence
    # mask : binary padded mask where 1's represent normal input and 0 represents masked input

    def __init__(self, positive, negative, tokenizer: BertTokenizer, max_len: int):
        super().__init__()
        self.max_len = max_len
        self.positive = positive if isinstance(positive, list) else [positive]
        self.negative = negative if isinstance(negative, list) else [negative]
        self.tokenizer = tokenizer
        self.labels = list()
        self.groups = list()
        self.sequences = list()
        self.process()


    def process(self):
        for label, files in [[1, self.positive],[0, self.negative]]:
            for fn in files: 
                bn = os.path.basename(fn)
                with open(fn) as infile:
                    for l in infile:
                        if l.startswith('ID_uniprot'):
                            continue
                        fields = l.strip().split(';')
                        if len(fields[1]) < 100:
                            seq = fields[2]
                        else:
                            seq = fields[1]
                        skip_left = (len(seq) - self.max_len) // 2
                        seq = seq[skip_left:skip_left + self.max_len]
                        self.sequences.append(seq)
                        self.groups.append(fields[0].split('_')[-1])
                        self.labels.append(label)
        self.labels = np.array(self.labels)
        self.groups = np.array(self.groups)
        self.sequences = np.array(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        label = int(self.labels[index])
        input_ids = torch.tensor(self.tokenizer.encode(' '.join(list(seq.upper()))))
        mask = torch.ones_like(input_ids)
        return input_ids, mask, label
    
    def __len__(self):
        return len(self.sequences)
    
    def collate_fn(self, inputs):
        ids, mask, label = map(list, zip(*inputs))
        ids = pad_sequence(ids, batch_first=True, 
                           padding_value=self.tokenizer.pad_token_id)
        mask = pad_sequence(mask, batch_first=True)
        label = torch.tensor(label)
        return ids, mask, label

# END OF CITATION 

# Code and comments below witten by Kevin Stull 
if __name__ == "__main__":

    ########### Testing the class #############

    # import form local dir 
    vocab_file = 'vocab.txt'

    # set the path to the folder of pre-trained SpliceBERT
    SPLICEBERT_PATH = os.getcwd() + "/SpliceBERT.510nt"

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)

    # Specify the directory path (DEPRECATED)
    positive_dir = os.getcwd() + '/spliceator/Training_data/Positive/GS'
    negative_dir = os.getcwd() + '/spliceator/Training_data/Negative/GS/GS_1'

    # List all files in the directory
    positive_files = [os.path.join(positive_dir, file) for file in os.listdir(positive_dir)]
    negative_files = [os.path.join(negative_dir, file) for file in os.listdir(negative_dir)]

    # Specify the maximum length
    max_len = 400

    # Create an instance of the SpliceatorDataset class
    dataset = SpliceatorDataset(
        positive=positive_files, 
        negative=negative_files, 
        tokenizer=tokenizer, 
        max_len=max_len
                                )

    # Access a few elements of the processed data
    labels = dataset.labels
    groups = dataset.groups
    sequences = dataset.sequences

    # Inspect the first 5 elements
    for i in range(22150,22155):
        print("Label:", labels[i])
        print("Group:", groups[i])
        print("Sequence:", sequences[i][:10])
        print()
    
    # Testing Out the helper functions 
    print("Num. Training Examples:", dataset.__len__())
    print("You can also do len(ds):", len(dataset))
    d = dataset.__getitem__(1)
    e = dataset.__getitem__(2)
    g = [d, e]

    # collate_fn makes batches from training examples
    print(np.shape(g))
    print(np.shape(dataset.collate_fn(g)))