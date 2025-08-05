import numpy as np
import pandas
from pathlib import Path

def compute_inference_metric():
    root = Path(__file__).parent
    data_folder = root / "data"
    data = pandas.read_csv(data_folder / "inference_times_stats_cpu_updated.csv")
    data = data.drop(['std'], axis=1)
    data['mean'] = np.exp(-data['mean'])
    data = data.sort_values(by=['mean'], ascending=False)
    data = data.rename(columns={'mean':'inference_metric'})
    return data

if __name__ == '__main__':
    print(compute_inference_metric())