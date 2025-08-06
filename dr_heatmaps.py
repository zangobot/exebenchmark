#%%

import multiprocessing

from evaluators.adv_evaluator import AdversarialEvaluator
from utils import read_json_file
import os
import time
import torch
from maltorch.adv.evasion.base_optim_attack_creator import OptimizerBackends
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

macro_config = read_json_file("configurations/ADVERSARIAL/adversarial.json")
thresholds = pd.read_csv("vanilla_thresholds.csv")
# Only use the first attack from the dictionary
attacks = dict(list(macro_config["attacks"].items()))
models = macro_config["models"][:6]
diagonal_scores_root = "adversarial_evaluation/adversarial_scores/"
other_scores_root = "adversarial_evaluation/transfer_scores/"
other_models = macro_config["models"][:6]

models = sorted(models)
# Custom sort: FDRS first, then KDRS, then RDRS, then SDRS
# def custom_sort(models_list):

#     fdrs = sorted([m for m in models_list if "FDRS" in m])
#     kdrs = sorted([m for m in models_list if "KDRS" in m])
#     rdrs = sorted([m for m in models_list if "RDRS" in m])
#     sdrs = sorted([m for m in models_list if "SDRS" in m])
#     return fdrs + kdrs + rdrs + sdrs

def custom_sort(models_list):

    rsdel = sorted([m for m in models_list if "RsDel" in m])
    rs = sorted([m for m in models_list if "RS" in m])
    return rsdel + rs

# other_models = custom_sort(other_models)

# Initialize a matrix to store the detection rates
heatmap_matrix = pd.DataFrame(
    0.0, index=other_models, columns=models
)

for attack, param in attacks.items():
    if "full_dos" in attack:
        attack = "full_dos"
    if "content_shift" in attack:
        attack = "content_shift"
    if param == "OptimizerBackends.NEVERGRAD":
        param = "nevergrad"
    if param == "OptimizerBackends.GRADIENT":
        param = "gradient"
    if attack == "gamma_5":
        attack = "gamma"
        param = "5"
    if attack == "gamma_10":
        attack = "gamma"
        param = "10"
    if attack == "gamma_20":
        attack = "gamma"
        param = "20"
    if attack == "gamma_30":
        attack = "gamma"
        param = "30"
    if attack == "gamma_50":
        attack = "gamma"
        param = "50"


    for j, model_d in enumerate(models):
        row = thresholds[thresholds.iloc[:, 0] == model_d]
        if row.empty:
            continue
        threshold_1 = row.iloc[0, 1]
        # threshold_1 = 0.5 

        # # take the diagonal scores
        diagonal_scores_path = os.path.join(diagonal_scores_root, model_d, attack + f"_{param}.csv")

        # if not os.path.exists(diagonal_scores_path):
        #     # Fill the entire column for this model_d with "-"
        #     heatmap_matrix.loc[:, model_d] = 0
        #     continue

        scores = pd.read_csv(diagonal_scores_path, header=None)
        score_values = scores.iloc[:, 1]
        detected = (score_values >= threshold_1).sum()
        total = len(score_values)

        dr = detected / total if total > 0 else 0.0

        # row_name = f"{model_d}_{attack}_{param}"

        # Set diagonal value
        heatmap_matrix.loc[model_d, model_d] = dr

        # other_models = [m for m in other_models if m != model_d]

        for i, model in enumerate(models):

            if model == model_d:
                continue

            row = thresholds[thresholds.iloc[:, 0] == model]
            # if row.empty:
            #     continue
            threshold_1 = row.iloc[0, 1]

            # threshold_1 = 0.5 

            attack_base = attack  # remove the last part of the attack name

            path = os.path.join(other_scores_root, model, attack_base, str(param), f"{model_d}.csv")

            if not os.path.exists(path):  
                heatmap_matrix.loc[:, model_d] = 0              
                continue

            scores = pd.read_csv(path, header=None)
            scores = scores.iloc[:, 0]

            detected_other = (scores >= threshold_1).sum()
            total_other = len(scores)

            dr_other = detected_other / total_other if total_other > 0 else 0.0

            # Set off-diagonal value (row: i, col: j)
            # heatmap_matrix.loc[model, model_d] = dr_other
            row_name = f"{model}_{attack}_{param}"
            # with open(f"dr_aggregate.csv", "a") as f:
            #     f.write(f"{row_name},{dr_other}\n")
            heatmap_matrix.loc[model, model_d] = dr_other

    # Save the heatmap matrix to a CSV file after each attack iteration
    # heatmap_matrix.to_csv(f"dr_aggregate.csv", mode='a')


    # import matplotlib.pyplot as plt

    # sns.set_style("white")
    # plt.rcParams.update({
    # "font.family": "DejaVu Serif",
    # "axes.labelsize": 16,
    # "axes.titlesize": 16,
    # "xtick.labelsize": 16,
    # "ytick.labelsize": 16,
    # "legend.fontsize": 16,
    # })

    # # Remove columns that are all zeros
    # heatmap_matrix = heatmap_matrix.loc[:, (heatmap_matrix != 0).any(axis=0)]

    # vmin, vmax = 0.0, 1.0

    # plt.figure(figsize=(9, 15))
    # ax = sns.heatmap(
    #     heatmap_matrix,
    #     annot=True,
    #     fmt=".2f",
    #     cmap="viridis",
    #     cbar_kws={
    #         'label': 'Detection Rate',
    #         'shrink': 1,
    #         'orientation': 'vertical',
    #         'pad': 0.1,
    #     },
    #     linewidths=0.5,
    #     vmax=1.0,
    #     vmin=0.0,
    #     square=True,
    #     annot_kws={"fontsize": 16}
    # )
    # cbar = ax.collections[0].colorbar
    # cbar.ax.yaxis.labelpad = 15
    # # plt.title(f"{attack}_{param} Detection Rates")
    # plt.gca().xaxis.set_label_position('top')
    # plt.gca().xaxis.tick_top()
    # plt.yticks(rotation=0)
    # plt.xticks(rotation=90)

    # # Highlight diagonal with thicker border
    # # for i in range(len(heatmap_matrix)):
    # #     ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='white', lw=3, clip_on=False))

    # ax.tick_params(axis='x', length=0)  # Rimuove barrette

    # cbar.remove()
    # plt.tight_layout()

    # plt.savefig(f"imgs/vanilla_certificate_{attack}_{param}.pdf", dpi=300, bbox_inches='tight')
    # plt.show()


# Now heatmap_matrix contains the detection rates for all (i, j) pairs
# You can save or plot it as needed, e.g.:
# heatmap_matrix.to_csv("heatmap_results.csv")








# %%
