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
import numpy as np
import os
import torch
import torch.nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, losses, models, InputExample
from transformers import AutoModel

class Pretrain_Dataset(Dataset):
    def __init__(self, original_dataset, max_seq_len, use_contrastive_learning=False):
        self.original_dataset = original_dataset
        self.max_seq_len = max_seq_len
        self.use_contrastive_learning = use_contrastive_learning
        self.data = self.generate_new_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def generate_new_dataset(self):
        new_dataset = []
        for a in self.original_dataset:
            start = np.random.randint(0, 100) # following convention of TNT
            for i in range(5900 // self.max_seq_len): # 6000 is max length of seq. in dataset 
                element = a['sequence'][start + i * self.max_seq_len: start + (i + 1) * self.max_seq_len]
                if 'N' not in element: # Throw out exmples containing N
                    if self.use_contrastive_learning: # saves as a tuple for contrastive learning 
                        new_dataset.append(InputExample(texts=[element, element]))
                    else: # save as a single element for MLM
                        new_dataset.append(InputExample(texts=[element]))
        return new_dataset


validation_dataset = load_from_disk("pretrain_hrg_validation.hf")
max_seq_len = 400
use_cl = True
pretrain_ds = Pretrain_Dataset(validation_dataset, max_seq_len, use_cl)
data_loader = torch.utils.data.DataLoader(pretrain_ds, batch_size=32, shuffle=True)

model_name = "SpliceBERT.510nt"  # Replace with the path to your custom model
hf_model = AutoModel.from_pretrained(model_name)
# Specify the maximum length
max_len = 400
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_len)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_loss = losses.MultipleNegativesRankingLoss(model)
num_epochs = 1
eval_steps = 100
model_save_path = os.getcwd() + '/exp1/'

# fit model 
model.fit(train_objectives=[(data_loader, train_loss)],
          epochs=num_epochs,
          evaluation_steps=eval_steps,
          output_path=model_save_path)
