# libraries 
import os
import sys
import logging
import argparse
import scanpy as sc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none' }
plt.rcParams.update(new_rc_params)
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

# files 
from utils import load_fasta, get_reverse_strand, encode_sequence, auto_open

# functions from fetch_embedding.py
ONEHOT = np.concatenate((
    np.zeros((1, 4)),
    np.eye(4),
)).astype(np.int8)

# not used 
def encode_dnabert(seq: str, k: int):
    seq_new = list()
    N = len(seq)
    seq = "N" * (k//2) + seq.upper() + "N" * k
    for i in range(k//2, N + k//2):
        seq_new.append(seq[i-k//2:i-k//2+k])
    return ' '.join(seq_new)

class FiexedBedData(Dataset):
    def __init__(self, bed, seq_len, genome, tokenizer=None, dnabert=None) -> None:
        super().__init__()
        self.dnabert = dnabert
        if dnabert is not None:
            assert dnabert in {3, 4, 5, 6}
        self.genome = load_fasta(genome)
        self.seq_len = seq_len
        self.bed = bed
        self.tokenizer = tokenizer
        self.samples = list()
        self.name = list()
        self.name2 = list()
        self.process()
    
    def process(self):
        with auto_open(self.bed, 'rt') as f:
            for line in f:
                chrom, start, end, name, name2, strand = line.strip().split('\t')
                start, end = int(start), int(end)
                left = (start + end) // 2 - self.seq_len // 2
                right = left + self.seq_len
                i = start - left
                j = end - left
                if strand == '-':
                    i, j = self.seq_len - 1 - j, self.seq_len - 1 - i
                self.samples.append((chrom, left, right, strand, i, j))
                self.name.append(name)
                self.name2.append(name2)
        self.samples = np.array(self.samples)
        self.name = np.array(self.name)
        self.name2 = np.array(self.name2)
    
    def __getitem__(self, index):
        chrom, left, right, strand, i, j = self.samples[index]
        left, right, i, j = int(left), int(right), int(i), int(j)
        seq = self.genome[chrom][left:right]
        if strand == '-':
            seq = get_reverse_strand(seq)
        if self.tokenizer is None:
            seq = torch.from_numpy(ONEHOT[encode_sequence(seq)])
        else:
            if self.dnabert is None:
                seq = torch.as_tensor(self.tokenizer.encode(' '.join(seq.upper())))
            else:
                seq = torch.as_tensor(self.tokenizer.encode(encode_dnabert(seq.upper(), self.dnabert)))
        return seq, i, j, self.name[index], self.name2[index]
    
    def __len__(self):
        return len(self.samples)
    
    def collate_fn(self, batch):
        seq, i, j, name, name2 = map(list, zip(*batch))
        seq = torch.stack(seq)
        i = np.asarray(i)
        j = np.asarray(j)
        name = np.asarray(name)
        name2 = np.asarray(name2)
        return seq, i, j, name, name2

# command line
parser = argparse.ArgumentParser(description='Generate UMAP data for evaluating models.')
parser.add_argument('--model', type=str, help='Path to the model')
# parser.add_argument('--bed', type=str, help='Path to the bed file')
parser.add_argument('--out_dir', type=str, help='Output directory')
parser.add_argument('--only_get_last_layer', action='store_true', help='Flag for only getting the last layer')
args = parser.parse_args()
model = args.model
# bed = args.bed
out_dir = args.out_dir
only_get_last_layer = args.only_get_last_layer

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = os.path.join(out_dir, "embed_data.txt")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# filepaths 
genome = '/home/data/hg19.fa'
bed = "/home/data/hg19.ss-motif.for_umap.bed.gz"
label = os.path.basename(bed)
output = out_dir + label

# arguments 
seed = 2023
skip_donor_acceptor_umap = True


# loads model 
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model, add_pooling_layer=False, output_hidden_states=True).to(device)
k = None 

# loads data 
ds = FiexedBedData(bed, 510, genome, tokenizer, dnabert=k)
batch_size = 16
loader = DataLoader(
    ds, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=ds.collate_fn
)

# calculates embeddings
embedding = list()
for it, (seq, i, j, name, name2) in enumerate(tqdm(loader)):
    seq = seq.to(device)
    h = model(seq).hidden_states
    h = torch.stack(h, dim=1).detach()
    del seq
    tmp_embed = list()
    for k in range(h.shape[0]):
        tmp_embed.append(h[k, :, i[k]+1:j[k]+1, :])
    embedding.append(torch.stack(tmp_embed, dim=0).detach().cpu().numpy().astype(np.float16))
embedding = np.concatenate(embedding, axis=0)
print("embedding shape:", embedding.shape, file=sys.stderr)

# choose which layers to calculate 
if only_get_last_layer:
    layers = [embedding.shape[1] - 1]
else:
    layers = range(embedding.shape[1])

# calculates chosen layers 
for h in tqdm(layers, desc="UMAP"):
    adata = sc.AnnData(embedding[:, h, :, :].reshape(embedding.shape[0], -1))
    adata.obs['name'] = ds.name
    adata.obs['name2'] = ds.name2
    adata.obs["label"] = [x.split("|")[-1] for x in ds.name]
    sc.pp.pca(adata, n_comps=128, random_state=0)
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata, random_state=0)
    sc.tl.leiden(adata)

    # logs data
    logger.info(f"{output}.L{h}.h5ad")
    logger.info(f"bed{bed}")

    # saves data
    adata.write_h5ad(f"{output}.L{h}.h5ad")

    # determines whether to skip AG/GT maps 
    if skip_donor_acceptor_umap:
        continue

    # calculates AG/GT maps 
    is_gt = np.asarray([x.split('|')[-1].startswith("GT") for x in adata.obs["label"]])
    is_ag = np.asarray([x.split('|')[-1].startswith("AG") for x in adata.obs["label"]])
    gt_adata = adata[is_gt]
    ag_adata = adata[is_ag]
    sc.pp.pca(gt_adata, n_comps=128, random_state=0)
    sc.pp.neighbors(gt_adata, use_rep='X_pca')
    sc.tl.umap(gt_adata, random_state=0)
    sc.tl.leiden(gt_adata)
    sc.pp.pca(ag_adata, n_comps=128, random_state=0)
    sc.pp.neighbors(ag_adata, use_rep='X_pca')
    sc.tl.umap(ag_adata, random_state=0)
    sc.tl.leiden(ag_adata)
    gt_adata.write_h5ad(f"{output}.L{h}.GT.h5ad")
    ag_adata.write_h5ad(f"{output}.L{h}.AG.h5ad")