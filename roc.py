#%%

import sys
import pandas as pd
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

def plot_multiple_roc(csv_paths):
    plt.figure(figsize=(10, 8))
    for path in csv_paths:
        df = pd.read_csv(path, header=None)
        y_true = df.iloc[:, 1].to_list()
        y_score = df.iloc[:, 0].to_list()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        filename = path.split('/')[-1].replace('.csv', '')
        label = f"{filename} (AUC = {roc_auc:.2f})"
        plt.plot(fpr, tpr, lw=2, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0001, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xscale('log')
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    paths = ["output/MalConv.csv", "output/NGramConv.csv", "output/EmberGBDT.csv"]
    plot_multiple_roc(paths)
# %%
