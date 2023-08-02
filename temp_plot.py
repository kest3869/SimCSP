import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

metrics = ['NMI', 'SCCS']
SCCS_scores = [0.0369, 0.832]
NMI_scores = [0.127, 0.150]
models = ['Pretrained', 'Fine-tuned']

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7,5))

# Set width of bar to be half of the total space for each x tick
barWidth = 1

# Set position of bar on X axis
r1 = np.arange(len(SCCS_scores))

# Make colormap that ranges from light to dark blue
cmap = plt.get_cmap('Blues')

# Calculate the middle range of the scores
middle_range_sccs = [np.mean(SCCS_scores) - np.std(SCCS_scores)/2, np.mean(SCCS_scores) + np.std(SCCS_scores)/2]
middle_range_nmi = [np.mean(NMI_scores) - np.std(NMI_scores)/2, np.mean(NMI_scores) + np.std(NMI_scores)/2]

# Normalize the scores to range between 0 and 1 for the colormap
norm_sccs = mcolors.Normalize(vmin=middle_range_sccs[0], vmax=middle_range_sccs[1])
norm_nmi = mcolors.Normalize(vmin=middle_range_nmi[0], vmax=middle_range_nmi[1])

# Make the plot for SCCS_scores
bars1 = axes[0].bar(r1, SCCS_scores, color=cmap(norm_sccs(SCCS_scores)), width=barWidth, edgecolor='grey', label='SCCS')
axes[0].set_xticks([r for r in range(len(SCCS_scores))])
axes[0].set_xticklabels(models)
axes[0].set_title('SCCS scores')

# Make the plot for NMI_scores
bars2 = axes[1].bar(r1, NMI_scores, color=cmap(norm_nmi(NMI_scores)), width=barWidth, edgecolor='grey', label='NMI')
axes[1].set_xticks([r for r in range(len(NMI_scores))])
axes[1].set_xticklabels(models)
axes[1].set_title('NMI scores')

# Adding labels 
for ax in axes:
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')

# Function to add value labels above the bars
def add_labels(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                '{:.3f}'.format(height),
                ha='center', va='bottom')

# Adding value labels to the bars
add_labels(bars1, axes[0])
add_labels(bars2, axes[1])

plt.tight_layout()
plt.subplots_adjust(top=0.90)  # Adjust the layout to make room for the title

OUT_DIR = '/storage/store/kevin/local_files/scrap/'

plt.savefig(OUT_DIR + 'score_comparison.png')
plt.close(fig)
