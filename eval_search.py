import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# Search_space
bss = [507, 2535, 4563, 6591, 8112, 10140, 12168]  # train steps (bs 512)

# Converts hyper-parameter to string associated with correct directory
hp_str = {
    507: '507',
    2535: '2535',
    4563: '4563',
    6591: '6591',
    8112: '8112',
    10140: '10140',
    12168: '12168'
}

# Create parent directory if it doesn't exist
parent_dir = "/home/search_space/hrg_val_test/"

# Create a list to store the boxplot data
boxplot_data = []

# Search loop
for bs in bss:
    # Construct OUT_DIR
    model_save_path = os.path.join(parent_dir, hp_str[bs])
    f1s = []
    # aucs = []

    # grab the metrics from each fold 
    for k in range(2):
        path = os.path.join(model_save_path, 'finetuned', 'fold' + str(k), 'val_metrics.pt')
        metrics = torch.load(path)
        f1 = metrics[0]
        # auc = metrics[1]
        f1s.append(f1)
        # aucs.append(auc)

    # Add the metrics to the boxplot data list
    boxplot_data.append(f1s)
    # boxplot_data.append(aucs)

# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 7))
# Plot the boxplots
bp = ax.boxplot(boxplot_data)
# Set the y-axis limits
ax.set_ylim(0.97, 1)
# Set the x-axis tick labels
ax.set_xticklabels(['F1'] * len(bss))
# Add labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('F1 and AUC Boxplots')
plt.savefig('SimCSPBoxplots.png', dpi=300)
plt.close(fig)
