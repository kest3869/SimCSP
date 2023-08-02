# imports 
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import torch 
from transformers import AdamW
import numpy as np
import os
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def chunk(sequence):
    chunks = []
    for seq in sequence["sequence"]:
        chunks += [seq[i:i + 510] for i in range(0, len(seq), 510)]
    return {"sequence": chunks}

def tokenize_function(examples):
    return tokenizer(examples['sequence'], truncation=True, padding='max_length', max_length=512)

OUT_DIR = "/storage/store/kevin/local_files/scrap"
model_path = "/storage/store/kevin/data/SpliceBERT-human.510nt/"
tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Load the training dataset
ds = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='train[:1%]').select(range(10))
print(f"Number of examples in training dataset: {len(ds)}")

# Preprocess the training dataset
ds = ds.map(chunk, remove_columns=ds.column_names, batched=True)
tokenized_ds = ds.map(tokenize_function, batched=True)
tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])

# Load the validation dataset
valid_ds = load_dataset('InstaDeepAI/human_reference_genome', '6kbp', split='validation[:1%]')
print(f"Number of examples in validation dataset: {len(valid_ds)}")

# Preprocess the validation dataset
valid_ds = valid_ds.map(chunk, remove_columns=valid_ds.column_names, batched=True)
tokenized_valid_ds = valid_ds.map(tokenize_function, batched=True)
tokenized_valid_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])

# Prepare the data loaders
train_loader = DataLoader(tokenized_ds, batch_size=2, shuffle=True, collate_fn=collator)
val_loader = DataLoader(tokenized_valid_ds, batch_size=8, collate_fn=collator)

# Print the shapes of the first batch in each loader
first_train_batch = next(iter(train_loader))
first_val_batch = next(iter(val_loader))
print(f"First train batch input shape: {first_train_batch['input_ids'].shape}")
print(f"First validation batch input shape: {first_val_batch['input_ids'].shape}")

# Load the model
model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
print("Model loaded successfully.")

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
scaler = GradScaler()

# Put the model in train mode
model.train()

# Clear the gradients
optimizer.zero_grad()

# Forward pass
outputs = model(first_train_batch)
print(f"Model output shape: {outputs.logits.shape}")

# Compute the loss
loss = outputs.loss
print(f"Loss: {loss.item()}")

# Backward pass
scaler.scale(loss).backward()

# Update weights
scaler.step(optimizer)
scaler.update()

# Put the model in evaluation mode
model.eval()
print("Model is now in evaluation mode.")

# Forward pass with a validation batch
val_output = model(first_val_batch)
print(f"Validation output shape: {val_output.logits.shape}")
