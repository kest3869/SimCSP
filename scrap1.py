# imports 
import argparse
import logging
import os
import gc
import sys 
import datetime
import torch
import torch.nn
from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset
from transformers import AdamW
from sentence_transformers import SentenceTransformer, InputExample, losses, models

# breaks the input data into 400 nucleotide sequences 
def chunk(data):
    chunks = []
    for seq in data["sequence"]:
        chunks += [seq[i:i + 510] for i in range(0, len(seq), 510)]
    return {"sequence": chunks}

# wraps Pyarrow dataset in Sentence Transformer class 
class InputDataset:
    def __init__(self,seq):
        self.seq = seq
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        return InputExample(texts=(self.seq[idx],self.seq[idx]))

OUT_DIR = '/storage/store/kevin/local_files/scrap/chop_layer/chopped_model/'
# PRETRAINED_MODEL = '/storage/store/kevin/data/SpliceBERT-human.510nt/'
NEW_MODEL = OUT_DIR + 'SpliceBERT-human.510nt/'
#os.mkdir(NEW_MODEL)

# hyperparameters
batch_size = 256 
learning_rate = 3e-5
max_seq_len = 510
use_cl = True 

# load cached dataset
ds = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='test')
# apply the mapping function to the dataset
ds = ds.map(chunk, remove_columns=ds.column_names, batched=True)
# make it compatible with Sentence Transformers library                            
ds = InputDataset(ds['sequence'])
# define dataloader
data_loader = DataLoader(ds,batch_size=batch_size,sampler=RandomSampler(ds), num_workers=4)

# define model 
word_embedding_model = models.Transformer(NEW_MODEL, max_seq_length=max_seq_len)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss=losses.MultipleNegativesRankingLoss(model)

# learning rate 
optimizer_class = AdamW
optimizer_params =  {'lr': learning_rate}

# fit model
model.fit(
    train_objectives=[(data_loader, train_loss)],
    epochs=1,
    optimizer_class=optimizer_class,
    optimizer_params=optimizer_params,
    output_path=OUT_DIR,
    use_amp = True,
    checkpoint_path = OUT_DIR + '/checkpoints/'
)

