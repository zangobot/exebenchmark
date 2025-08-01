from pathlib import Path

import pandas

def compute_performance_metric():
    root = Path(__file__).parent
    data_folder = root / "data"
    data = pandas.read_csv(data_folder / "metrics_aggregate.csv")
    data = data.rename(columns={"f1_score":"performance_metric"})
    data = data.drop(["accuracy","TPR","FPR"], axis=1)
    data = data.sort_values(by="performance_metric", ascending=False)
    return data
