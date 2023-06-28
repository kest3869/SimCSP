
# libraries 
import torch 
import matplotlib.pyplot as plt

f1, auc = list(), list()
for i in range(10):
    path = 'results/fold' + str(i) + '/test_metrics.pt'
    metrics = torch.load(path)
    f1.append(metrics[0])
    auc.append(metrics[1])

fig, ax = plt.subplots(figsize=(10, 7))
# Create a list of the four datasets
data = [f1, auc]
# Plot the boxplots
bp = ax.boxplot(data)
# Set the y-axis limits
ax.set_ylim(0.97, 1)
# Set the x-axis tick labels
ax.set_xticklabels(['F1', 'AUC'])
# Add labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('F1 and AUC Boxplot')
plt.savefig('SpliceBERTBoxplot.png', dpi=300)
plt.close(fig)

