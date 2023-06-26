from tqdm import tqdm
import torch
from datasets import load_from_disk
import numpy as np
from torch.utils.data import Dataset

ds = load_from_disk('/home/pretrain_hrg_validation.hf')

print(ds) # have to use subset
