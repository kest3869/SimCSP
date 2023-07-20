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
from torch.utils.data import Subset 
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# files
import split_spliceator
import load

# Environment 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# global variables 
patience = 2 # if set to 1, will terminate training as soon as validation accuracy does not increase 
time_since_last_val_improvement = 0 
best_score = -1

# logs callback information during pre-training and performs early stopping
def callbacks(score, epoch, steps):

    global time_since_last_val_improvement 
    global best_score
    global patience 

    # save information in logger 
    print("Score from model.fit call")
    print({'score':score, 'epoch':epoch, 'steps':steps})


    if score > best_score:
        best_score = score
        time_since_last_val_improvement = 0
        print("Validation accuracy has improved!")
        print("best_score", best_score)
    else:
        time_since_last_val_improvement += 1
        print("validation accuracy has not improved")
        print("time since last val improvement", time_since_last_val_improvement)
        print("best_score", best_score)
    # end training if run out of patience 
    if time_since_last_val_improvement >= patience:
        print("early stopping at steps", steps)
        print("patience: ", patience)
        sys.exit()
    return None

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

OUT_DIR = "/storage/store/kevin/SimCSP/scrap/"
SPLIT_DIR = "/storage/store/kevin/local_files/exp1/SPLITS/"

# model and dataset for train
ds = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='train')
# apply the mapping function to the dataset
ds = ds.map(chunk, remove_columns=ds.column_names, batched=True)
# make it compatible with Sentence Transformers library                            
ds = InputDataset(ds['sequence'])
# hyperparameters
batch_size = 256
learning_rate = 3e-5 
max_seq_len = 510
wd = 1E-6
use_cl = True 
# define dataloader
data_loader = DataLoader(ds,batch_size=batch_size,sampler=RandomSampler(ds), num_workers=4)
# define model 
model_path = "/storage/store/kevin/data/SpliceBERT-human.510nt"  
# build model for contrasitve learning
word_embedding_model = models.Transformer(model_path, max_seq_length=max_seq_len)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss=losses.MultipleNegativesRankingLoss(model)
# number of epochs 
num_epochs = 2
# learning rate 
optimizer_class = AdamW
optimizer_params =  {'lr': learning_rate,
                     'weight_decay': wd}

# evaluator and dataset for eval
tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')
# Positive and Negative paths
positive_dir = '/storage/store/kevin/data/spliceator/Training_data/Positive/GS'
negative_dir = '/storage/store/kevin/data/spliceator/Training_data/Negative/GS/GS_1'
# List all files in the directory
positive_files = [os.path.join(positive_dir, file) for file in os.listdir(positive_dir)]
negative_files = [os.path.join(negative_dir, file) for file in os.listdir(negative_dir)]
# Load dataset using class from load.py file
ds_eval = load.SpliceatorDataset(
    positive=positive_files,
    negative=negative_files,
    tokenizer=tokenizer,
    max_len=400
)
# load the split the fine-tuned model used 
validation_split = torch.load(SPLIT_DIR + '/test_split.pt')
# call split_spliceator.pre_val_data to build semantic similarity dataset from subset for only first fold
ds_eval = split_spliceator.prep_val_data(Subset(ds_eval,validation_split[0]), tokenizer)
# create our evaluator (only using the first fold for sccs)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(ds_eval, 
                                                            batch_size=64, 
                                                            name='spliceator_test_split',
                                                            show_progress_bar=True)

# fit model
model.fit(
    train_objectives=[(data_loader, train_loss)],
    epochs=num_epochs,
    evaluator=evaluator,
    optimizer_class=optimizer_class,
    optimizer_params=optimizer_params,
    output_path= OUT_DIR + '/pretrained_models/',
    use_amp = True,
    save_best_model = True,
    evaluation_steps = 100,
    checkpoint_path = OUT_DIR+'/pretrained_models/checkpoints/',
    callback = callbacks,
    checkpoint_save_steps = 100,
    checkpoint_save_total_limit = patience + 2,
)


