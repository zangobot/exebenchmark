from functools import reduce
from pathlib import Path

import pandas

from results.adv.robustness_metric import compute_robustness_metric
from results.inference.inference_metric import compute_inference_metric
from results.performance.performance_metric import compute_performance_metric
from results.spie_chart import create_spie_charts
from results.temporal.temporal_metric import compute_temporal_metric

def compute_benchmark():
    robustness = compute_robustness_metric()
    temporal = compute_temporal_metric()
    inference = compute_inference_metric()
    performance = compute_performance_metric()

    frames = [performance, inference, temporal, robustness]

    results = reduce(lambda l, r: pandas.merge(l, r, on=["model"]), frames)
    results['rank'] = (results['performance_metric'] + results['temporal_metric'] + results['robustness_metric'] + results[
        'inference_metric']) / 4
    results = results.sort_values(by="rank", ascending=False)
    results.to_csv(Path(__file__).parent / "ranking.csv")
    create_spie_charts()
    return results

if __name__ == '__main__':
    print(compute_benchmark())