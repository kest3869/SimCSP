import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

models = ['Baseline', 'SCCS', 'NMI']
species = ['Danio', 'Fly', 'Thaliana', 'Worm']
f1_base = [0.9466, 0.9278, 0.9218, 0.9007]
f1_sccs = [0.9432, 0.9229, 0.9130, 0.8914]
f1_nmi = [0.9437, 0.9223, 0.9123, 0.8893]
f1_scores = [f1_base, f1_sccs, f1_nmi]

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=1, ncols=len(species), figsize=(12,5))

# Set width of bar to be half of the total space for each x tick
barWidth = 0.5

# Make colormap that ranges from light to dark blue
cmap = plt.get_cmap('Blues')

for i, ax in enumerate(axes):
    # Prepare scores for a particular species
    species_scores = [f1_scores[j][i] for j in range(len(models))]
    # Calculate the middle range of the scores
    middle_range = [np.mean(species_scores) - np.std(species_scores)/2, np.mean(species_scores) + np.std(species_scores)/2]
    # Normalize the scores to range between 0 and 1 for the colormap
    norm = mcolors.Normalize(vmin=middle_range[0], vmax=middle_range[1])
    # Make the plot for each model
    bars = ax.bar(models, species_scores, color=cmap(norm(species_scores)), width=barWidth, edgecolor='grey')
    ax.set_title(species[i])

    # Adding labels 
    ax.set_xlabel('Models')
    ax.set_ylabel('F1 Scores')
    
    # Set y limits
    ax.set_ylim([0.85, 0.97])

    # Function to add value labels above the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    '{:.4f}'.format(height),
                    ha='center', va='bottom')
    # Adding value labels to the bars
    add_labels(bars)

plt.tight_layout()
plt.subplots_adjust(top=0.90)  # Adjust the layout to make room for the title

OUT_DIR = '/storage/store/kevin/local_files/scrap/'
plt.savefig(OUT_DIR + 'score_comparison.png')
plt.close(fig)
