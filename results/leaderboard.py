from functools import reduce
from pathlib import Path
import numpy as np
import pandas

from adv.robustness_metric import compute_robustness_metric
from inference.inference_metric import compute_inference_metric
from performance.performance_metric import compute_performance_metric
from temporal.temporal_metric import compute_temporal_metric

def compute_benchmark(mean=True):
    robustness = compute_robustness_metric()
    temporal = compute_temporal_metric()
    inference = compute_inference_metric()
    performance = compute_performance_metric()

    frames = [performance, inference, temporal, robustness]

    results = reduce(lambda l, r: pandas.merge(l, r, on=["model"]), frames)
    if mean:
        results['rank'] = (results['performance_metric'] + results['temporal_metric'] + results['robustness_metric'] +
                           results[
                               'inference_metric']) / 4
        results_file = "ranking_mean.csv"
    else:
        results['rank'] = results['rank'] = results[
            ['performance_metric', 'temporal_metric', 'robustness_metric', 'inference_metric']].median(axis=1)

        results_file = "ranking_median.csv"

    results = results.sort_values(by="rank", ascending=False)
    results.to_csv(Path(__file__).parent / results_file)
    return results

if __name__ == '__main__':
    results = compute_benchmark()