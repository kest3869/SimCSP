# imports 
import os
import logging
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import EvalPrediction
from transformers import EarlyStoppingCallback
from torch.nn import functional as F
import torch

# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# breaks the input data into 400 nucleotide sequences 
def chunk(sequence):
    chunks = []
    for seq in sequence["sequence"]:
        chunks += [seq[i:i + 510] for i in range(0, len(seq), 510)]
    return {"sequence": chunks}

# tokenizes the dataset 
def tokenize(sequence):
    return tokenizer(sequence["sequence"])

# compute accuracy of MLM
def compute_metrics(p: EvalPrediction):
    logits = p.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = torch.tensor(logits).to(device)  # Convert logits to PyTorch tensor
    labels = torch.tensor(p.label_ids).to(device)  # Convert labels to PyTorch tensor

    # Create a mask to ignore pad tokens and non-masked tokens
    mask = (labels != tokenizer.pad_token_id) & (labels != tokenizer.mask_token_id)

    # Get predictions
    predictions = torch.argmax(logits, dim=-1)  # Get model's predictions

    # Calculate accuracy
    correct_predictions = (predictions == labels) * mask  # Count only correct predictions where mask is True
    accuracy = correct_predictions.sum().float() / mask.sum().float()

    # Return accuracy
    return {"accuracy": accuracy.item()}

# args
OUT_DIR = "/storage/store/kevin/local_files/scrap"
model_path = "/storage/store/kevin/data/SpliceBERT-human.510nt/"  
tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')
batch_size_train = 512
learning_rate = 5e-5 
wd = 1e-6
num_epochs = 2
max_seq_len = 510
mlm_prob = 0.15
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(OUT_DIR + 'pretrain.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# prepare datasets
ds = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='train[:1%]').select(range(10))
ds = ds.map(chunk, remove_columns=ds.column_names, batched=True)
ds = ds.map(tokenize, batched=True)
ds_valid = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='validation[:10%]').select(range(10))
ds_valid = ds_valid.map(chunk, remove_columns=ds_valid.column_names, batched=True)
ds_valid = ds_valid.map(tokenize, batched=True)

# define model for MLM
model_MLM = AutoModelForMaskedLM.from_pretrained(model_path)
data_collator = DataCollatorForLanguageModeling(tokenizer)

# initialize trainer 
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    weight_decay=wd,
    fp16=True,
    optim='adamw_torch',
    learning_rate=learning_rate,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    per_device_train_batch_size=batch_size_train,
    logging_dir=OUT_DIR,
    save_total_limit=1,
    load_best_model_at_end=True,  
    metric_for_best_model="accuracy",  
    greater_is_better=False,  
)
trainer = Trainer(
    model=model_MLM,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds_valid,
    data_collator=data_collator, 
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# evaluate the model 
trainer.train()

# save the metadata for later reference 
metadata = {
    "model_path": model_path,
    "learning_rate": learning_rate,
    "weight_decay": wd,
    "num_epochs": num_epochs,
    "batch_size_train": batch_size_train,
    "max_seq_len": max_seq_len,
    "mlm_prob": mlm_prob,
}
logger.info('Finished with hyperparameters: %s', metadata)
