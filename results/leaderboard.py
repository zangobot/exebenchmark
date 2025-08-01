from pathlib import Path

import pandas

root = Path(__file__).parent

def load_result_dataframe(result_path: Path):
    csv_file = pandas.read_csv(result_path)
    return csv_file

def time_degradation(model_name):
    ...

def robustness_degradation(model_name, attack_suffix_name):
    ...

def _tr_requirements():
    tr_path = root / "training_times.csv"
    tr_time_df = load_result_dataframe(tr_path)
    worst_tr = max(tr_time_df['total_training_time_seconds'])
    result_df = tr_time_df[["model", "total_training_time_seconds"]]
    result_df["tr_ratio"] = 1 - (result_df["total_training_time_seconds"]/worst_tr)
    return result_df

def _ts_requirements():
    tr_path = root / "inference_times_stats_cpu.csv"
    tr_time_df = load_result_dataframe(tr_path)
    worst_tr = max(tr_time_df['mean'])
    result_df = tr_time_df[["model", "mean"]]
    result_df["ts_ratio"] = 1 - (result_df["mean"]/worst_tr)
    return result_df

def computational_requirements():
    tr_cpu = _tr_requirements()
    ts_cpu = _ts_requirements()
    results = pandas.merge(tr_cpu, ts_cpu)
    results['cpu_req_metric'] = (results['tr_ratio'] + results['ts_ratio']) / 2
    return results

print(computational_requirements()[['model', 'tr_ratio', 'ts_ratio']])

