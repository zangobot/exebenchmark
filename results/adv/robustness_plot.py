import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from results.adv.robustness_metric import compute_robustness_metric
from results.constants import REGULAR_MODELS, CERTIFICATION_MODELS, MODELS

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
cmap = plt.cm.viridis
markers = ['.', 'v', '^', 's', 'x', '|']
colors = cmap(np.linspace(0, 1, len(REGULAR_MODELS)+1))

data_path = Path(__file__).parent / "plot_data.pickle"
if not data_path.exists():
    compute_robustness_metric()
with open(str(data_path), 'rb') as h:
    plot_data = pickle.load(h)

# x = plot_data['eps'][:-1]
x = np.arange(stop=1, step=1/len(plot_data['eps'][:-1]))
# plt.figure()
# for m in MODELS:
#     plt.semilogx(x, np.array(plot_data[m]) / 4764, label=m) 
# plt.xlabel("eps (bytes)", fontsize=18)
# plt.ylabel("DR", fontsize=18)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.legend(fontsize=13)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10,6))
for i, m in enumerate(REGULAR_MODELS):
    plt.plot(x, np.array(plot_data[m]) / 4764, label=m, linewidth=4, color=colors[i])
plt.xlabel("Perturbation (bytes)", fontsize=18)
plt.ylabel("Detection Rate", fontsize=18)
plt.legend()
plt.grid(True, alpha=0.5, linewidth=2)
plt.tight_layout()
# plt.savefig("robustness_dr_eps.pdf")
plt.show()
# plt.figure(figsize=(10,3))
# for m in CERTIFICATION_MODELS:
#     plt.semilogx(x, np.array(plot_data[m]) / 4764, label=m)
# plt.xlabel("eps (bytes)", fontsize=18)
# plt.ylabel("DR", fontsize=18)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.legend(fontsize=13)
# plt.tight_layout()
# plt.show()