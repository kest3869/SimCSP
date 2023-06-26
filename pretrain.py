# CITATION
'''
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
'''

# imports 
import numpy as np
import os
import argparse
import sys 
from tqdm import tqdm
import datetime
import torch
import torch.nn
from torch.utils.data import Dataset
from transformers import AdamW
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, losses, models, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

class Prepare_Dataset(Dataset):
    def __init__(self, original_dataset, max_seq_len, use_contrastive_learning=False):
        self.original_dataset = original_dataset
        self.max_seq_len = max_seq_len
        self.use_contrastive_learning = use_contrastive_learning
        self.new_dataset = list()
        self.data = self.generate_new_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def generate_new_dataset(self):
        for a in self.original_dataset:
            start = np.random.randint(0, 100) # following convention of TNT
            for i in range(5900 // self.max_seq_len): # 6000 is max length of seq. in dataset 
                element = a['sequence'][start + i * self.max_seq_len: start + (i + 1) * self.max_seq_len]
                if 'N' not in element: # Throw out exmples containing N
                    if self.use_contrastive_learning: # saves as a tuple for contrastive learning 
                        self.new_dataset.append(InputExample(texts=[element, element]))
                    else: # save as a single element for MLM
                        self.new_dataset.append(InputExample(texts=[element]))
        return self.new_dataset

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
model_save_path = args.model_save_path

# Check if the command line arguments are provided
if batch_size is None or learning_rate is None or model_save_path is None:
    parser.error('Missing command line argument(s), -b batch_size, -l learning_rate, -p out/path/for/trained/model')

# load data
max_seq_len = 400
use_cl = True
ds = tqdm(load_from_disk('/home/pretrain_hrg_validation.hf'), desc='Loading Raw Train Data')
pretrain_ds = Prepare_Dataset(ds, max_seq_len, True)
del ds
data_loader = torch.utils.data.DataLoader(pretrain_ds, batch_size=batch_size, shuffle=True)

# define model 
model_path = "/home/SpliceBERT.510nt/"  
word_embedding_model = models.Transformer(model_path, max_seq_length=max_seq_len)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss=losses.MultipleNegativesRankingLoss(model)

# define validation model 
ds_val = tqdm(load_from_disk('/home/pretrain_hrg_test.hf'), desc="Loading Raw Test Data")
validation_ds = Prepare_Dataset(ds_val, max_seq_len, True)
del ds_val
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_ds, batch_size=8, name='hrg-val')
dev_evaluator(model)

# number of epochs 
num_epochs = 15

# learning rate 
optimizer_class = AdamW
optimizer_params =  {'lr': learning_rate}

# fit model
model.fit(
    train_objectives=[(data_loader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=100,
    optimizer_class=optimizer_class,
    optimizer_params=optimizer_params,
    output_path=model_save_path + 'pretrained_model_scrap/'
)

# Mark training as finished
with open(model_save_path + 'finished_pretrain.txt', 'w') as file:
    file.write(str(datetime.datetime.now().time()))
