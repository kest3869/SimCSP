# libraries 
import os
import logging
import argparse
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none' }
plt.rcParams.update(new_rc_params)
from sklearn.metrics import normalized_mutual_info_score

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
parser.add_argument('--data', type=str, help='Path to the model')
parser.add_argument('--layer', type=str, help='Layer being plotted')
parser.add_argument('--out_dir', type=str, help='Output directory')
args = parser.parse_args()
DATA = args.data
OUT_DIR = args.out_dir
layer=args.layer

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = os.path.join(OUT_DIR, "embed_plot.log")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# metric used 
metric_fun = normalized_mutual_info_score

# generates plot
splicebert_ss = sc.read_h5ad(DATA)
sc.pl.umap(splicebert_ss, color=["label", "leiden"])
args = dict(show=False, legend_loc="right margin", s=1)
loc = (-0.01, 1)
mark_loc = (0.5, 0.07)
mark_size = 8
fig, ax = plt.subplots()
ax.text(*loc, "A", transform=ax.transAxes, fontsize=20, fontweight='normal', va='top', ha='right')
sc.pl.umap(splicebert_ss, color=["label"], ax=ax, title="SimCSP", **args)
nmi_score, nmi_donor, nmi_acceptor = cal_metric_by_group(splicebert_ss.obs["label"], splicebert_ss.obs["leiden"], metric_fun)
ax.text(*mark_loc, f"NMI(GT/AG)={nmi_donor:.2f}/{nmi_acceptor:.2f}", transform=ax.transAxes, fontsize=mark_size, fontweight='normal', va='top', ha='center')
ax.set_xlabel("")
ax.set_ylabel("")
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# save plot metrics to logger
logger.info(f"nmi_score{nmi_score},nmi_donor{nmi_donor},nmi_accepter{nmi_acceptor}")
logger.info(DATA)
logger.info(OUT_DIR)

# save plot to file 
plt.savefig(os.path.join(OUT_DIR, f"NMI_plot_layer_{layer}"), dpi=600, bbox_inches="tight")
