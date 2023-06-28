from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample
import split_spliceator  # splits the dataset for validation during pre-training 
from transformers import AutoTokenizer


# load cached dataset
ds = load_dataset('InstaDeepAI/human_reference_genome', split='validation')

# breaks the input data into 400 nucleotide sequences 
def chunk(data):
    chunks = []
    for seq in data["sequence"]:
        chunks += [seq[i:i + 400] for i in range(0, len(seq), 400)]
    return {"sequence": chunks}

# apply the mapping function to the dataset
ds = ds.map(chunk, remove_columns=ds.column_names, num_proc=8, batched=True)

# wraps Pyarrow dataset in Sentence Transformer class 
class InputDataset:
    def __init__(self,seq):
        self.seq = seq
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        return InputExample(texts=(self.seq[idx],self.seq[idx]))

# make it compatible with Sentence Transformers library                            
ds = InputDataset(ds['sequence'])

####### TESTING WITH SimCSP CODE ##########
# imports 
from tqdm import tqdm
import datetime
import torch
import torch.nn
from torch.utils.data import Dataset
from transformers import AdamW
from sentence_transformers import SentenceTransformer, losses, models, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# hyperparameters
batch_size = 512 # chosen by expermiment 
learning_rate = 3e-5 # chosen by epxeriment 
model_save_path = '/home/scrap/'
max_seq_len = 400
use_cl = True 
eval_bs = 64 # turn this down during experiments
eval_per_epoch = 2 # number of times to evaluate the model per epoch

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

eval_steps = len(ds) // (batch_size * 1) # evaluated the data at the end of every epoch

# number of epochs 
num_epochs = 3

# learning rate 
optimizer_class = AdamW
optimizer_params =  {'lr': learning_rate}

# fit model
model.fit(
    train_objectives=[(data_loader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=eval_steps,
    optimizer_class=optimizer_class,
    optimizer_params=optimizer_params,
    output_path=model_save_path + 'pretrained_model_scrap/'
)

# Mark training as finished
with open(model_save_path + 'finished_pretrain.txt', 'w') as file:
    file.write(str(datetime.datetime.now().time()))