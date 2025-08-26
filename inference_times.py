#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the file
df = pd.read_csv('results/raw_results/inference_times_cpu_updated.csv')

# Remove the first column
df = df.iloc[:, 1:]

# Create a boxplot for each column using matplotlib
plt.figure(figsize=(14, 9))
plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 16,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })
df_melted = df.melt(var_name='Model', value_name='Inference Time')
sns.boxplot(x='Model', y='Inference Time', data=df_melted, 
                showfliers=False, 
                showmeans=True,    
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black", "markersize": 8, "marker": "o"},
                width=0.7,
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='black'),


)
plt.ylabel('Inference Time (s)')
plt.yscale('log')
plt.xticks(rotation=90)
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('')
plt.tight_layout()
plt.show()

