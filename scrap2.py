# libraries 
import os
import logging
import argparse
import gc
import scanpy as sc
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

# files 
from utils import load_fasta, get_reverse_strand, auto_open
from get_paths import get_paths

# environment 
new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none' }
plt.rcParams.update(new_rc_params)

# functions from fetch_embedding.py
ONEHOT = np.concatenate((
    np.zeros((1, 4)),
    np.eye(4),
)).astype(np.int8)

# load embed dataset 
class FiexedBedData(Dataset):
    def __init__(self, bed, seq_len, genome, tokenizer=None, dnabert=None) -> None:
        super().__init__()
        self.dnabert = dnabert
        if dnabert is not None:
            assert dnabert in {3, 4, 5, 6}
        self.genome = load_fasta(genome)
        self.seq_len = seq_len
        self.bed = bed
        self.tokenizer = tokenizer
        self.samples = list()
        self.name = list()
        self.name2 = list()
        self.process()
    
    def process(self):
        with auto_open(self.bed, 'rt') as f:
            for line in f:
                chrom, start, end, name, name2, strand = line.strip().split('\t')
                start, end = int(start), int(end)
                left = (start + end) // 2 - self.seq_len // 2
                right = left + self.seq_len
                i = start - left
                j = end - left
                if strand == '-':
                    i, j = self.seq_len - 1 - j, self.seq_len - 1 - i
                self.samples.append((chrom, left, right, strand, i, j))
                self.name.append(name)
                self.name2.append(name2)
        self.samples = np.array(self.samples)
        self.name = np.array(self.name)
        self.name2 = np.array(self.name2)
    
    def __getitem__(self, index):
        chrom, left, right, strand, i, j = self.samples[index]
        left, right, i, j = int(left), int(right), int(i), int(j)
        seq = self.genome[chrom][left:right]
        if strand == '-':
            seq = get_reverse_strand(seq)
        seq = torch.as_tensor(self.tokenizer.encode(' '.join(seq.upper())))

        return seq, i, j, self.name[index], self.name2[index]
    
    def __len__(self):
        return len(self.samples)
    
    def collate_fn(self, batch):
        seq, i, j, name, name2 = map(list, zip(*batch))
        seq = torch.stack(seq)
        i = np.asarray(i)
        j = np.asarray(j)
        name = np.asarray(name)
        name2 = np.asarray(name2)
        return seq, i, j, name, name2

# filepaths 
genome = '/storage/store/kevin/data/hg19.fa'
bed = "/storage/store/kevin/data/hg19.ss-motif.for_umap.bed.gz"
label = os.path.basename(bed)
tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')

# arguments 
seed = 2023
skip_donor_acceptor_umap = True
dev = True
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loads data 
ds = FiexedBedData(bed, 510, genome, tokenizer)
batch_size = 8
loader = DataLoader(
    ds, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=ds.collate_fn
)

print(len(ds))
