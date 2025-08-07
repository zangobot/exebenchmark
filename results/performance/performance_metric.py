from pathlib import Path

import pandas
from sklearn.metrics import f1_score
from results.constants import MODELS


def compute_performance_metric():
    root = Path(__file__).parent
    data_folder = root / "data"
    data = pandas.read_csv(data_folder / "metrics_aggregate.csv")
    data = data.rename(columns={"f1_score":"performance_metric"})
    data = data.drop(["accuracy","TPR","FPR"], axis=1)
    data = data.sort_values(by="performance_metric", ascending=False)
    return data

def compute_performance_metric_f1_all():
    root = Path(__file__).parent.parent.parent / "output" / "testset"
    results = {}
    for m in MODELS:
        data = pandas.read_csv(f"{root / m}.csv", header=None)
        y_pred = data[0] >= 0.5
        y_true = data[1]
        f1score = f1_score(y_true, y_pred)
        results[m] = f1score
    data = pandas.DataFrame(data=[[k, results[k]] for k in MODELS], columns=["model", "performance_metric"]).sort_values(by="performance_metric", ascending=False)
    return data

if __name__ == '__main__':
    print(compute_performance_metric())