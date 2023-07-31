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
import numpy as np
import torch
import torch.nn
from torch.utils.data import DataLoader, RandomSampler, Subset
from datasets import load_dataset
from transformers import AdamW, AutoTokenizer
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator

# files
import split_spliceator
import load

# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# breaks the input data into 400 nucleotide sequences 
def chunk(data):
    chunks = []
    for seq in data["sequence"]:
        chunks += [seq[i:i + 510] for i in range(0, len(seq), 510)]
    return {"sequence": chunks}


# logs callback information during pre-training and performs early stopping
def callbacks(score, epoch, steps):
    # save information in logger 
    logger.info("Score")
    logger.info({'score':score, 'epoch':epoch, 'steps':steps})
    return None

# wraps huggingface dataset in Sentence Transformer compatible class 
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
parser.add_argument('--model_save_path', type=str, help='The model save path')
parser.add_argument('--pretrained_model_path', type=str, help='The pre-trained model path')
parser.add_argument('--batch_size', type=int, help='The batch size used')
parser.add_argument('--learning_rate', type=float, help='The learning rate used')
parser.add_argument('--weight_decay', type=float, help='the parameter used for weight decay')
parser.add_argument('--split_dir', type=str, help='The path to the splits')
# Parse the command line arguments
args = parser.parse_args()
# Retrieve the values of the command line arguments
OUT_DIR = args.model_save_path
PRETRAINED_MODEL_PATH = args.pretrained_model_path
SPLIT_DIR = args.split_dir
batch_size = args.batch_size
learning_rate = args.learning_rate
wd = args.weight_decay

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

# load dataset
ds = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='train')
# apply the mapping function to the dataset
ds = ds.map(chunk, remove_columns=ds.column_names, batched=True)
# make it compatible with Sentence Transformers library                            
ds = InputDataset(ds['sequence'])

# hyperparameters
max_seq_len= 510
num_epochs= 1
num_steps= 250
eval_bs= 8
# define dataloader
data_loader = DataLoader(ds,batch_size=batch_size,sampler=RandomSampler(ds), num_workers=4)

# build model for contrasitve learning
word_embedding_model = models.Transformer(PRETRAINED_MODEL_PATH, max_seq_length=max_seq_len)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss=losses.MultipleNegativesRankingLoss(model)
# learning rate 
optimizer_class = AdamW
optimizer_params =  {'lr': learning_rate,
                     'weight_decay': wd}

# load dataset for evaluators
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
split = torch.load(SPLIT_DIR + '/train_split.pt') 

# create an evaluator for each fold of the training data 
evaluator_list = []
for i in range(np.shape(split)[0]):
    # call split_spliceator.pre_val_data to build semantic similarity dataset from subset for only first fold
    ds_prepped = split_spliceator.prep_val_data(Subset(ds_eval, split[i]), tokenizer)
    # create our evaluator (only using the first fold for sccs)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(ds_prepped,
                                                                batch_size=eval_bs, 
                                                                name='spliceator_train_split',
                                                                show_progress_bar=True)
    evaluator_list.append(evaluator)

# define a sequential evaluator
seq_eval = SequentialEvaluator(evaluator_list, np.mean)

# fit model
model.fit(
    train_objectives=[(data_loader, train_loss)],
    epochs=num_epochs,
    evaluator=seq_eval,
    optimizer_class=optimizer_class,
    optimizer_params=optimizer_params,
    output_path= OUT_DIR + '/pretrained_models/',
    use_amp = True,
    evaluation_steps = num_steps,
    checkpoint_path = OUT_DIR+'/pretrained_models/checkpoints/',
    save_best_model = True,
    callback = callbacks,
    checkpoint_save_steps = num_steps,
    checkpoint_save_total_limit = 100,
)

# Save hyperparameter info to the logger
metadata = {
    'learning_rate': learning_rate,
    'weight_decay' : wd,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'max_seq_ln' : max_seq_len,
    'evaluation step size' : num_steps,
    'evaluation batch size' : eval_bs,
    'optimizer': 'AdamW',
    'loss' : 'MultipleNegativeRankings',
    'tokenizer used' : '/storage/store/kevin/data/tokenizer_setup',
    'split_dir' : SPLIT_DIR,
    'base model': PRETRAINED_MODEL_PATH,
    'outdir': OUT_DIR,
    'pretrained_model': OUT_DIR + '/pretrained_models/',
    'number train examples:' : len(ds),
    'number eval examples:': len(ds_eval),
    'finished at time': datetime.datetime.now().time()
}
logger.info('Finished with hyperparameters: %s', metadata)