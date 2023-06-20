from datasets import load_dataset

# loading pretraining data  
dataset_name = 'InstaDeepAI/human_reference_genome'
# Load the dataset
dataset = load_dataset(dataset_name)
# save to disk 
# Save each split to disk
for split in dataset.keys():
    dataset[split].save_to_disk(f"pretrain_hrg_{split}.hf")