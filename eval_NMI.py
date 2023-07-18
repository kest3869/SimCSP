# libraries 
import os
import sys
import logging
import argparse
import scanpy as sc
import torch
import datetime
import numpy as np
import csv
from sklearn.metrics import normalized_mutual_info_score

# files
from get_paths import get_paths

# caluclates metrics
def cal_metric_by_group(labels, preds, metric_fun, by_group: bool=True):
    if by_group:
        k1 = np.isin(labels, ['GT(donor)', "GT(non-donor)", 'donor(SSE < 0.2)', 'donor(SSE > 0.8)'])
        score1 = metric_fun(labels[k1], preds[k1])
        k2 = np.isin(labels, ['AG(acceptor)', "AG(non-acceptor)", 'acceptor(SSE < 0.2)', 'acceptor(SSE > 0.8)'])
        score2 = metric_fun(labels[k2], preds[k2])
        return (score1 + score2) / 2, score1, score2
    else:
        return metric_fun(labels, preds)
    
# command line
parser = argparse.ArgumentParser(description='Generate UMAP plots for evaluating models.')
parser.add_argument('--layer', type=str, help='Layer being plotted')
parser.add_argument('--out_dir', type=str, help='Output directory')
args = parser.parse_args()
OUT_DIR = args.out_dir
layer=args.layer

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = os.path.join(OUT_DIR + '/results/', "NMI_results.log")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# check for completion of evaluation
if os.path.exists(OUT_DIR + '/results/' + '/finished_eval_NMI.pt'):
    logger.info("Found finished: Skipping eval_NMI.py")
    print("Found finished, skipping eval_NMI.py!")
    sys.exit()
# metric used 
metric_fun = normalized_mutual_info_score

# paths
bed = "/storage/store/kevin/data/hg19.ss-motif.for_umap.bed.gz"
label = os.path.basename(bed)
pretrain_paths, finetune_paths = get_paths(OUT_DIR)
pretrain_NMI_scores = []
pretrain_NMI_paths = []
finetune_NMI_scores = []
finetune_NMI_paths = []

# looping through pretrained paths and generating NMI scores
for path in pretrain_paths:
    # builds path to pre-computed embedding
    path = path + '/' + label + '.L6.h5ad'

    # generate metrics from data
    splicebert_ss = sc.read_h5ad(path)
    nmi_score, nmi_donor, nmi_acceptor = cal_metric_by_group(splicebert_ss.obs["label"], splicebert_ss.obs["leiden"], metric_fun)

    # save metrics
    logger.info(f"nmi_score{nmi_score},nmi_donor{nmi_donor},nmi_accepter{nmi_acceptor}")
    logger.info(path)
    pretrain_NMI_scores.append(nmi_score)
    pretrain_NMI_paths.append(path)

# looping through finetuned paths and generating NMI scores
for path in finetune_paths:

    # only use the first fold of fine-tune
    if 'fold0' not in path:
        continue

    # builds path to pre-computed embedding
    path = path + '/' + label + '.L6.h5ad'

    # generate metrics from data
    splicebert_ss = sc.read_h5ad(path)
    nmi_score, nmi_donor, nmi_acceptor = cal_metric_by_group(splicebert_ss.obs["label"], splicebert_ss.obs["leiden"], metric_fun)

    # save metrics
    logger.info(f"nmi_score{nmi_score},nmi_donor{nmi_donor},nmi_accepter{nmi_acceptor}")
    logger.info(path)
    finetune_NMI_scores.append(nmi_score)
    finetune_NMI_paths.append(path)

# save CSV of pretrained NMI scores
pretrain_data = [["path", "score"]]
for path, score in zip(pretrain_NMI_paths, pretrain_NMI_scores):
    pretrain_data.append([path, score])
with open(OUT_DIR + '/results/' + 'pretrain_NMI_results.csv', "w", newline="") as file1:
    writer = csv.writer(file1)
    writer.writerows(pretrain_data)

# save CSV of finetuned NMI scores
finetune_data = [["path", "score"]]
for path, score in zip(finetune_NMI_paths, finetune_NMI_scores):
    finetune_data.append([path, score])
with open(OUT_DIR + '/results/' + 'finetune_NMI_results.csv', "w", newline="") as file2:
    writer = csv.writer(file2)
    writer.writerows(finetune_data)

# mark evaluation as finished
torch.save(datetime.datetime.now().time(), OUT_DIR + '/results/' + '/finished_eval_NMI.pt')