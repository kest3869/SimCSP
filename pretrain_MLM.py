# imports 
import os
import logging
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

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

# args
OUT_DIR = "/storage/store/kevin/local_files/BASELINE/SpliceBERT-HUMAN_MLM/"
model_path = "/storage/store/kevin/data/SpliceBERT-human.510nt/"  
tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')
batch_size_train = 512
learning_rate = 5e-5 
wd = 1e-6
num_epochs = 10
max_seq_len = 510
mlm_prob = 0.15

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(OUT_DIR + '/pretrained_models/' + 'pretrain.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# prepare datasets
ds = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='train')
ds = ds.map(chunk, remove_columns=ds.column_names, batched=True)
ds = ds.map(tokenize, batched=True)

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
    per_device_train_batch_size=batch_size_train,
    logging_dir=OUT_DIR,
    save_total_limit=1
)
trainer = Trainer(
    model=model_MLM,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator, 
)

# train the model 
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
