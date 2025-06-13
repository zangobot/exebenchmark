#%%

import sys
import pandas as pd
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

def find_threshold(fpr, tpr, thresholds, target_fpr):
    """
    Returns the threshold, FPR, and TPR for the FPR closest to target_fpr.
    If not found, returns (None, None, None).
    """
    import numpy as np
    if len(fpr) == 0:
        return None, None, None
    idx = np.where(fpr >= target_fpr)[0]
    if len(idx) == 0:
        return None, None, None
    i = idx[0]  # first index where fpr >= target_fpr
    return thresholds[i], fpr[i], tpr[i]

def plot_multiple_roc(csv_paths):
    plt.figure(figsize=(10, 8))
    threshold_results = []
    desired_fpr = 10**-2  # set your desired FPR value here (0.01)

    for path in csv_paths:
        df = pd.read_csv(path, header=None)
        y_true = df.iloc[:, 1].to_list()
        y_score = df.iloc[:, 0].to_list()
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        filename = path.split('/')[-1].replace('.csv', '')
        label = f"{filename} (AUC = {roc_auc:.2f})"
        plt.plot(fpr, tpr, lw=3, label=label)
        # Use find_threshold to get threshold at closest FPR to desired_fpr
        threshold, found_fpr, found_tpr = find_threshold(fpr, tpr, thresholds, desired_fpr)
        print(f"Threshold: {threshold}")
        print(f"FPR: {found_fpr}")
        print(f"TPR: {found_tpr}")
        threshold_results.append((filename, threshold))

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0001, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xscale('log')
    plt.xlabel('False Positive Rate (log scale)', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curves', fontsize=18)
    plt.legend(loc="lower right", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # Save thresholds to CSV
    import csv
    with open('roc_thresholds1.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', f'threshold_at_fpr_{desired_fpr}'])
        writer.writerows(threshold_results)

if __name__ == "__main__":
    import os
    testset_dir = "output/testset"
    paths = [os.path.join(testset_dir, f) for f in os.listdir(testset_dir) if f.endswith(".csv")]
    # paths = [
    #     "output/testset/AvastStyleConv.csv",
    #     "output/testset/BBDnn.csv",
    #     "output/testset/MalConv.csv",
    #     "output/testset/NGramConv.csv",
    #     "output/testset/EmberGBDT.csv",
    #     "output/testset/ResNet18.csv",
    # ]
    plot_multiple_roc(paths)
# %%
