# libraries 
import numpy as np
from transformers import AutoTokenizer

# files 
import split_spliceator  # splits the dataset for validation during pre-training 

# load datasets 
model_path = "/home/SpliceBERT.510nt/"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
ds_val = split_spliceator.split_spliceator(True, tokenizer)
ds_prepped = split_spliceator.prep_val_data(ds_val, tokenizer)


# prints 0.5040315274506251
labels = []
for i in range(len(ds_val)): 
    _, _, label = ds_val[i]
    labels.append(label)

print("% pos spliceator labels:", np.sum(labels) / len(labels))

# prints 0.5028990759195506
labels = []
for example in ds_prepped: 
    label = example.label
    labels.append(label)

print("% pos validation labels:", np.sum(labels) / len(labels))
