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
import sys 
import datetime
import torch
import torch.nn
from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset
from transformers import AutoTokenizer, AdamW
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import SentenceTransformer, losses, models, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# files 
import split_spliceator  # splits the dataset for validation during pre-training 

# breaks the input data into 400 nucleotide sequences 
def chunk(data):
    chunks = []
    for seq in data["sequence"]:
        chunks += [seq[i:i + 400] for i in range(0, len(seq), 400)]
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
#parser.add_argument('-b', '--batch_size', type=int, help='The batch size')
#parser.add_argument('-l', '--learning_rate', type=float, help='The learning rate')
parser.add_argument('-p', '--model_save_path', type=str, help='The model save path')

# Parse the command line arguments
args = parser.parse_args()

# Retrieve the values of the command line arguments
#batch_size = args.batch_size
#learning_rate = args.learning_rate
OUT_DIR = args.model_save_path

# Check if the command line arguments are provided
if OUT_DIR is None:
    parser.error('Missing command line argument(s), -p out/path/for/trained/model')

# make path
PRETRAINED_MODEL = OUT_DIR + 'pretrained/'
# make directory if it does not exist 
if not os.path.exists(PRETRAINED_MODEL):
    os.makedirs(PRETRAINED_MODEL)
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(OUT_DIR + 'pretrain.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# skip if already completed 
if os.path.exists(PRETRAINED_MODEL + 'finished.pt'):
    logger.info("Found Pretrained: Skipping Pretrain")
    sys.exit()

# start time 
st = datetime.datetime.now().time()
# save start time of fitting 
logger.info("start time:{st}".format(st=st))

# load cached dataset
ds = load_dataset('InstaDeepAI/human_reference_genome', split='train')
# apply the mapping function to the dataset
ds = ds.map(chunk, remove_columns=ds.column_names, num_proc=4, batched=True)
# make it compatible with Sentence Transformers library                            
ds = InputDataset(ds['sequence'])

# hyperparameters
batch_size = 512 # chosen by expermiment 
learning_rate = 3e-5 # chosen by experiment 
max_seq_len = 400
use_cl = True 
eval_bs = 8 # turn this down during experiments

# define dataloader
data_loader = DataLoader(ds,batch_size=batch_size,sampler=RandomSampler(ds))

# define model 
model_path = "/home/SpliceBERT.510nt/"  
word_embedding_model = models.Transformer(model_path, max_seq_length=max_seq_len)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss=losses.MultipleNegativesRankingLoss(model)

# define an evaluator 
tokenizer = AutoTokenizer.from_pretrained(model_path)
ds_val = split_spliceator.split_spliceator(True, tokenizer)
ds_prepped = split_spliceator.prep_val_data(ds_val, tokenizer)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(ds_prepped, 
                                                             batch_size=eval_bs, 
                                                             name='spliceator_pretrain_split',
                                                             show_progress_bar=True)
eval_steps = len(ds) // batch_size # evaluate the data at the end of every epoch

# number of epochs 
num_epochs = 25

# learning rate 
optimizer_class = AdamW
optimizer_params =  {'lr': learning_rate}

# fit model
model.fit(
    train_objectives=[(data_loader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    optimizer_class=optimizer_class,
    optimizer_params=optimizer_params,
    output_path=PRETRAINED_MODEL,
    use_amp = True,
    callback = callbacks,
    checkpoint_path = PRETRAINED_MODEL,
    checkpoint_save_steps = eval_steps,
    checkpoint_save_total_limit = num_epochs*2,
)

# Save hyperparameter info to the logger
metadata = {
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': 1,
    'optimizer': 'AdamW',
    'base model': model_path,
    'loss':'MultipleNegativeRankings',
    'len(val_ds)': len(ds_val),
    'bs_val': eval_bs,
    'outdir':OUT_DIR,
    'pretrained_model':PRETRAINED_MODEL,
    'number examples:':len(ds),
    'time':datetime.datetime.now().time()
}

# mark training as finished
torch.save(datetime.datetime.now().time(), OUT_DIR + 'finished.pt')
logger.info('Finished with hyperparameters: %s', metadata)
