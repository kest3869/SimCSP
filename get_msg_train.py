# libraries
from datasets import load_dataset

# name of dataset
dataset_name = 'InstaDeepAI/multi_species_genomes'
# Load dataset
msg_train = load_dataset(dataset_name, split='train')
# display metadata
print(msg_train)