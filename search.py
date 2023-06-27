
# Libraries
import argparse
import torch
import torch.nn
from sentence_transformers import InputExample
from datasets import load_dataset

# Files 
import search_helpers

# Create the argument parser
parser = argparse.ArgumentParser(description='Pretrain model')
parser.add_argument('-e', '--epochs', type=int, help='The number of epochs')
parser.add_argument('-o', '--out_dir', type=str, help='The model save path')

# Parse the command line arguments
args = parser.parse_args()

# Retrieve the values of the command line arguments
epochs = args.batch_size
OUT_DIR = args.model_save_path

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths  
PRETRAINED_MODEL = '/home/SpliceBERT.510nt'

# Load data first 
max_seq_len = 400
use_cl = True
# load cached dataset
ds = load_dataset('InstaDeepAI/multi_species_genomes', split='train[1%:5%]')
# breaks the input data into 400 nucleotide sequences 
def chunk(data):
    chunks = []
    for seq in data["sequence"]:
        chunks += [seq[i:i + 400] for i in range(0, len(seq), 400)]
    return {"sequence": chunks}
# apply the mapping function to the dataset
ds = ds.map(chunk, remove_columns=ds.column_names, num_proc=8, batched=True)
# wraps dataset in Sentence Transformer class 
class InputDataset:
    def __init__(self,seq):
        self.seq = seq
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, idx):
        return InputExample(texts=(self.seq[idx],self.seq[idx]))
# make it compatible with Sentence Transformers library                            
ds = InputDataset(ds['sequence'])

# Pre-training
PRETRAINED_MODEL = search_helpers.pretrain_model(ds, OUT_DIR, epochs)

# Fine-tuning 
# search_helpers.finetune_model(PRETRAINED_MODEL, OUT_DIR)

