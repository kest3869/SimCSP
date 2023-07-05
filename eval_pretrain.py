import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
path = '/home/search_space/hrg_val_test/512/3e5/'
df = pd.read_csv(path + 'pretrained/eval/similarity_evaluation_spliceator_pretrain_split_results.csv')

# Set the x-axis as the epoch column
x = df['epoch']

# Set the y-axis for each metric
cosine_pearson = df['cosine_pearson']
cosine_spearman = df['cosine_spearman']
manhattan_pearson = df['manhattan_pearson']
manhattan_spearman = df['manhattan_spearman']
dot_pearson = df['dot_pearson']
dot_spearman = df['dot_spearman']

# Create the line graphs
plt.plot(x, cosine_pearson, label='Cosine Pearson')
plt.plot(x, cosine_spearman, label='Cosine Spearman')
plt.plot(x, manhattan_pearson, label='Manhattan Pearson')
plt.plot(x, manhattan_spearman, label='Manhattan Spearman')
plt.plot(x, dot_pearson, label='Dot Pearson')
plt.plot(x, dot_spearman, label='Dot Spearman')

# Set the title, x-axis label, and y-axis label
plt.title('Metric Evolution')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')

# Add a legend
plt.legend()

# Specify the output directory and filename
output_dir = path
filename = 'line_graphs.png'

# Save the line graph to the specified output directory
plt.savefig(f'{output_dir}/{filename}')

# Close the plot to free up resources
plt.close()