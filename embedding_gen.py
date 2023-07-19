# libraries 
import os
import logging
import argparse
import gc
import scanpy as sc
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

# files 
from utils import load_fasta, get_reverse_strand, auto_open
from get_paths import get_paths

# environment 
new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none' }
plt.rcParams.update(new_rc_params)

# functions from fetch_embedding.py
ONEHOT = np.concatenate((
    np.zeros((1, 4)),
    np.eye(4),
)).astype(np.int8)

# load embed dataset 
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
        seq = torch.as_tensor(self.tokenizer.encode(' '.join(seq.upper())))

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
parser.add_argument('--out_dir', type=str, help='Output directory')
parser.add_argument('--only_get_last_layer', action='store_true', help='Flag for only getting the last layer')
args = parser.parse_args()
OUT_DIR = args.out_dir
only_get_last_layer = args.only_get_last_layer

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = os.path.join(OUT_DIR, "embed_data.log")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# filepaths 
genome = '/storage/store/kevin/data/hg19.fa'
bed = "/storage/store/kevin/data/hg19.ss-motif.for_umap.bed.gz"
label = os.path.basename(bed)
tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')

# arguments 
seed = 2023
skip_donor_acceptor_umap = True
dev = True
max_pretrain_computed = 15 # the maximum number of checkpoints computed (pre-training only)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loads data 
ds = FiexedBedData(bed, 510, genome, tokenizer)
batch_size = 16
loader = DataLoader(
    ds, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=ds.collate_fn
)

# list of filepaths in outdir 
_, finetune_paths = get_paths(OUT_DIR)
# go through and calculate embeddings
for path in finetune_paths: 


    # only compute first fold of fine-tune
    if 'fold0' not in path:
        continue

    # skip if already completed 
    if os.path.exists(path + '/finished_embed_gen.pt'):
        logger.info("Found embed: Skipping embed_gen")
        continue

    # get our unqiue mdel
    output = path + '/' + label 
    model = AutoModel.from_pretrained(path, add_pooling_layer=False, output_hidden_states=True).to(device)

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

    # choose which layers to cluster 
    if only_get_last_layer:
        layers = [embedding.shape[1] - 1]
    else:
        layers = range(embedding.shape[1])

    # calculates clusters 
    for h in tqdm(layers, desc="UMAP"):
        adata = sc.AnnData(embedding[:, h, :, :].reshape(embedding.shape[0], -1))
        adata.obs['name'] = ds.name
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

    # collection of metadata from embed_gen
    metadata = {
        'out_dir' : path,
        'model_path' : model,
        'only_get_last_layer' : only_get_last_layer,
        'genome' : genome,
        'bed' : bed,
        'seed' : seed,
        'skip_donor_acceptor_umap' : skip_donor_acceptor_umap
    }

    # mark evaluation as finished
    torch.save(datetime.datetime.now().time(), path + '/finished_embed_gen.pt')
    logger.info('Finished finetuned with hyperparameters: %s', metadata)
    gc.collect()

computed = 0
# list of filepaths in outdir 
pretrain_paths, _ = get_paths(OUT_DIR)
# go through and calculate embeddings
for path in pretrain_paths: 

    # stop don't compute the rest of the embeddings
    if computed >= max_pretrain_computed:
        continue

    # skip if already completed 
    if os.path.exists(path + '/finished_embed_gen.pt'):
        logger.info("Found embed: Skipping embed_gen")
        continue

    # get our unqiue mdel
    output = path + '/' + label 
    model = AutoModel.from_pretrained(path, add_pooling_layer=False, output_hidden_states=True).to(device)

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

    # choose which layers to cluster 
    if only_get_last_layer:
        layers = [embedding.shape[1] - 1]
    else:
        layers = range(embedding.shape[1])

    # calculates clusters 
    for h in tqdm(layers, desc="UMAP"):
        adata = sc.AnnData(embedding[:, h, :, :].reshape(embedding.shape[0], -1))
        adata.obs['name'] = ds.name
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
        
    # increment computed
    computed += 1  

    # collection of metadata from embed_gen
    metadata = {
        'out_dir' : path,
        'model_path' : model,
        'max_pretrain_computed' : max_pretrain_computed,
        'only_get_last_layer' : only_get_last_layer,
        'genome' : genome,
        'bed' : bed,
        'seed' : seed,
        'skip_donor_acceptor_umap' : skip_donor_acceptor_umap
    }

    # mark training as finished
    torch.save(datetime.datetime.now().time(), path + '/finished_embed_gen.pt')
    logger.info('Finished pretrained with hyperparameters: %s', metadata)
    gc.collect()
