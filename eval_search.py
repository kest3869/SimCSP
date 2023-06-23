import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# Search_space
bss = [512, 256, 64]  # batch sizes
lrs = [1e-5, 3e-5, 5e-5]  # learning rates

# Converts hyper-parameter to string associated with correct directory
hp_str = {
    64: '64',
    256: '256',
    512: '512',
    1e-5: '1e5',
    3e-5: '3e5',
    5e-5: '5e5'
}

# Create parent directory if it doesn't exist
parent_dir = "/home/search_space/hrg_test/"

# Create a list to store the boxplot data
boxplot_data = []

# Search loop
for bs in bss:
    for lr in lrs:
        # Construct OUT_DIR
        model_save_path = os.path.join(parent_dir, hp_str[bs], hp_str[lr])
        f1s = []
        aucs = []

        # grab the metrics from each fold 
        for k in range(10):
            path = os.path.join(model_save_path, 'finetuned', 'fold' + str(k), 'val_metrics.pt')
            metrics = torch.load(path)
            f1 = metrics[0]
            auc = metrics[1]
            f1s.append(f1)
            aucs.append(auc)

        # Add the metrics to the boxplot data list
        boxplot_data.append(f1s)
        boxplot_data.append(aucs)

# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 7))
# Plot the boxplots
bp = ax.boxplot(boxplot_data)
# Set the y-axis limits
ax.set_ylim(0.97, 1)
# Set the x-axis tick labels
ax.set_xticklabels(['F1', 'AUC'] * len(bss) * len(lrs))
# Add labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('F1 and AUC Boxplots')
plt.savefig('Boxplots.png', dpi=300)
plt.close(fig)

# Create a list to store the mean F1 scores
mean_f1s = []

# Search loop
for bs in bss:
    for lr in lrs:
        # Construct OUT_DIR
        model_save_path = os.path.join(parent_dir, hp_str[bs], hp_str[lr])
        f1s = []

        # grab the F1 scores from each fold 
        for k in range(10):
            path = os.path.join(model_save_path, 'finetuned', 'fold' + str(k), 'val_metrics.pt')
            metrics = torch.load(path)
            f1 = metrics[0]
            f1s.append(f1)

        # Calculate the mean F1 score and add it to the list
        mean_f1 = np.mean(f1s)
        mean_f1s.append(mean_f1)

# Create the chart
fig, ax = plt.subplots(figsize=(10, 7))
x_pos = np.arange(len(mean_f1s))
bar_width = 0.35

# Plot the bars
bars = ax.bar(x_pos, mean_f1s, bar_width)

# Set the x-axis tick labels
ax.set_xticks(x_pos)
ax.set_xticklabels(['F1, AUC'] * len(bss) * len(lrs))

# Add labels and title
ax.set_xlabel('Combination of F1 and AUC')
ax.set_ylabel('Mean F1 Score')
ax.set_title('Mean F1 Score for Each Combination of F1 and AUC')

# Add mean F1 score values above each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('Mean_F1_Scores.png', dpi=300)
plt.close(fig)