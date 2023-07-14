from datasets import load_dataset, concatenate_datasets
import torch

OUT_DIR1 = '/home/data/train'
OUT_DIR2 = '/home/data/val_test'
dataset = 'InstaDeepAI/human_reference_genome'

# load dataset
train = load_dataset(dataset, split='train')
val = load_dataset(dataset, split='validation')
test = load_dataset(dataset, split='test')
# breaks the input data into 510 nucleotide sequences 
def chunk(data):
    chunks = []
    for seq in data["sequence"]:
        chunks += [seq[i:i + 510] for i in range(0, len(seq), 510)]
    return {"sequence": chunks}

train = train.map(chunk, remove_columns=train.column_names, batched=True)
val = val.map(chunk, remove_columns=val.column_names, batched=True)
test = test.map(chunk, remove_columns=test.column_names, batched=True)
val_test = concatenate_datasets([val, test])
train.save_to_disk(OUT_DIR1, num_proc=16)
val_test.save_to_disk(OUT_DIR2)
