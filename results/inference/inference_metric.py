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

def collect_training_time():
    root = Path(__file__).parent
    data_folder = root / "data" / "training_times.csv"
    data = pandas.read_csv(data_folder)
    data['tr_hour'] = (data['total_training_time_seconds'] + data['preprocessing_time_seconds'])  / 3600
    data = data[['model','tr_hour']]
    data = data.sort_values(by="tr_hour", ascending=True)
    return data

if __name__ == '__main__':
    print(compute_inference_metric())
    print(collect_training_time())