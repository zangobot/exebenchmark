import matplotlib.pyplot as plt

from results.constants import REGULAR_MODELS
from results.temporal.temporal_metric import load_temporal_data, year_quarters

data, total_temporal_samples = load_temporal_data()
plt.figure()
for m in REGULAR_MODELS:
    f1_m = [data[data['model'] == m][f"F1-Score@{yq}"] for yq in year_quarters[:-2]]
    plt.plot(f1_m, label=m)
plt.ylabel("F1 Score")
plt.xticks(ticks=list(range(len(year_quarters[:-2]))), labels=year_quarters[:-2])
plt.legend()
plt.show()