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

# logs callback information during pre-training
def callbacks(score, epoch, steps):
    # save information in logger 
    logger.info({'score':score, 'epoch':epoch, 'steps':steps})
    return None

# Create the argument parser
parser = argparse.ArgumentParser(description='Pretrain model')
parser.add_argument('-p', '--model_save_path', type=str, help='The model save path')
# Parse the command line arguments
args = parser.parse_args()
# Retrieve the values of the command line arguments
OUT_DIR = args.model_save_path
# make directory if it does not exist 
if not os.path.exists(OUT_DIR + '/pretrained_models/checkpoints/'):
    os.makedirs(OUT_DIR + '/pretrained_models/checkpoints/')
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(OUT_DIR + '/pretrained_models/' + 'pretrain.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# skip if already completed 
if os.path.exists(OUT_DIR + '/pretrained_models/' + 'finished_pretrain.pt'):
    logger.info("Found Pretrained Model: Skipping Pretrain")
    sys.exit()

# start time 
st = datetime.datetime.now().time()
# save start time of fitting 
logger.info("start time:{st}".format(st=st))

# load cached dataset
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
num_epochs = 1

# learning rate 
optimizer_class = AdamW
optimizer_params =  {'lr': learning_rate,
                     'weight_decay': wd}

# fit model
model.fit(
    train_objectives=[(data_loader, train_loss)],
    epochs=num_epochs,
    optimizer_class=optimizer_class,
    optimizer_params=optimizer_params,
    output_path= OUT_DIR + '/pretrained_models/',
    use_amp = True,
    callback = callbacks,
    checkpoint_path = OUT_DIR+'/pretrained_models/checkpoints/',
    checkpoint_save_steps = 500,
    checkpoint_save_total_limit = 1000,
)

# Save hyperparameter info to the logger
metadata = {
    'learning_rate': learning_rate,
    'weight_decay' : wd,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'optimizer': 'AdamW',
    'base model': model_path,
    'loss':'MultipleNegativeRankings',
    'outdir':OUT_DIR,
    'pretrained_model': OUT_DIR + '/pretrained_models/',
    'number examples:':len(ds),
    'finished at time':datetime.datetime.now().time()
}

# mark training as finished
torch.save(datetime.datetime.now().time(), OUT_DIR + '/pretrained_models/' + 'finished_pretrain.pt')
logger.info('Finished with hyperparameters: %s', metadata)
