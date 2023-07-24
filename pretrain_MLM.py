# imports 
import argparse
import logging
import os
import sys 
import datetime
import numpy as np
import torch
import torch.nn
from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset
from transformers import AdamW
from torch.utils.data import Subset 
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, PrinterCallback
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator

# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

# breaks the input data into 400 nucleotide sequences 
def chunk(data):
    chunks = []
    for seq in data["sequence"]:
        chunks += [seq[i:i + 510] for i in range(0, len(seq), 510)]
    return {"sequence": chunks}

# args
OUT_DIR = "/storage/store/kevin/SimCSP/scrap/"
batch_size = 512
learning_rate = 1e-4 
wd = 1e-6
num_epochs = 3
max_seq_len = 510
mlm_prob = 0.15
patience = 1
model_path = "/storage/store/kevin/data/SpliceBERT-human.510nt/"  

# load train dataset
ds = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='train[:1%]')
# apply the mapping function to the dataset
ds = ds.map(chunk, remove_columns=ds.column_names, batched=True)
# express as a np.array for sentence_transformers 
ds_df = ds.to_pandas()
ds_list = ds_df['sequence'].tolist()
ds_prepped = np.array(ds_list)
del ds
del ds_df
del ds_list

# load validation dataset 
ds_val = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='validation[:10%]')
ds_val = ds_val.map(chunk, remove_columns=ds_val.column_names, batched=True)
ds_val_df = ds_val.to_pandas()
ds_val_list = ds_val_df['sequence'].tolist()
ds_val_prepped = np.array(ds_val_list)
del ds_val
del ds_val_df
del ds_val_list

# define model for MLM
model_MLM = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

# wrap datasets for sentence transformers 
ds_MLM = TokenizedSentencesDataset(ds_prepped, tokenizer, max_seq_len)
ds_MLM_val = TokenizedSentencesDataset(ds_val_prepped, tokenizer, max_seq_len)

# args for trainer
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    do_train=False,
    do_eval=True,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    weight_decay=wd,
    fp16=True,
    optim='adamw_torch',
    learning_rate=learning_rate,
    dataloader_num_workers=4,
    load_best_model_at_end = True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=batch_size,
    logging_dir='scrap_tf_log',
    save_total_limit=1
)

# initialize trainer
trainer = Trainer(
    model=model_MLM,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds_MLM,
    eval_dataset=ds_MLM_val,
    callbacks=[PrinterCallback, EarlyStoppingCallback(patience)]
)

# train model 
print(trainer.evaluate())