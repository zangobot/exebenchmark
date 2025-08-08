import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scienceplots
from results.adv.robustness_metric import compute_robustness_metric
from results.constants import REGULAR_MODELS, CERTIFICATION_MODELS, MODELS
plt.style.use(['science','no-latex', 'high-vis'])

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

data_path = Path(__file__).parent / "plot_data.pickle"
ranking_path = Path(__file__).parent.parent / "ranking.csv"
if not data_path.exists():
    compute_robustness_metric()
with open(str(data_path), 'rb') as h:
    plot_data = pickle.load(h)
data = pandas.read_csv(ranking_path)
top_six_models = data['model'][:6].tolist()
x = plot_data['eps'][:-1]

plt.figure(figsize=(10,6))
for i, m in enumerate(top_six_models):
    plt.semilogx(x, np.array(plot_data[m]) / 4764, label=m, linewidth=4)
plt.xlabel("Perturbation (bytes)")
plt.ylabel("Detection Rate")
plt.legend()
plt.grid(True, alpha=0.5, linewidth=2)
plt.tight_layout()
plt.savefig("robustness_dr_eps.pdf")
