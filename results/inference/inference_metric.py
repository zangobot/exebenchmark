import numpy as np
import pandas
from pathlib import Path

root = Path(__file__).parent
data_folder = root / "data"

def compute_inference_metric():
    data = pandas.read_csv(data_folder / "inference_times_stats_cpu_updated.csv")
    data = data.drop(['std'], axis=1)
    data['mean'] = np.exp(-data['mean'])
    data = data.sort_values(by=['mean'], ascending=False)
    data = data.rename(columns={'mean':'inference_metric'})
    return data