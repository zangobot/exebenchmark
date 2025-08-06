from pathlib import Path

import pandas

root = Path(__file__).parent
data_folder = root / "data"

years = list(range(2019, 2023))
quarters = [f"Q{i}" for i in range(1, 4)]
year_quarters = []

for y in years:
    for q in quarters:
        year_quarters.append(f"{y}-{q}")


def _parse_temporal_data():
    temporal_data = pandas.read_csv(data_folder / "temporal_aggregate.csv")
    temporal_data = temporal_data.rename(columns={'n_malware': 'n_malware.0', 'n_goodware': 'n_goodware.0'})

    malware_to_rename = {
        f'n_malware.{i}': f'n_malware.{yq}' for i, yq in enumerate(year_quarters)
    }
    malware_to_rename['n_malware.10'] = 'n_malware.other_future'
    malware_to_rename['n_malware.11'] = 'n_malware.past'
    goodware_to_rename = {
        f'n_goodware.{i}': f'n_goodware.{yq}' for i, yq in enumerate(year_quarters)
    }
    malware_to_rename['n_goodware.10'] = 'n_goodware.other_future'
    malware_to_rename['n_goodware.11'] = 'n_goodware.past'
    temporal_data = temporal_data.rename(columns=malware_to_rename)
    temporal_data = temporal_data.rename(columns=goodware_to_rename)
    return temporal_data


def compute_temporal_metric():
    data, total_temporal_samples = load_temporal_data()
    temporal_metric = {}
    for m in data['model']:
        temporal_metric[m] = 0
        for yq in year_quarters[:-2]:
            f1_yq = data[data['model'] == m][f"F1-Score@{yq}"]
            n_malware = data[data['model'] == m][f"n_malware.{yq}"]
            n_goodware = data[data['model'] == m][f"n_goodware.{yq}"]
            sample_per_yq = n_malware + n_goodware
            weight = sample_per_yq / total_temporal_samples
            temporal_metric[m] += weight.item() * f1_yq.item()
    temporal_metric = pandas.DataFrame(data=[[m, temporal_metric[m]] for m in data['model']],
                                       columns=['model', 'temporal_metric']).sort_values(by='temporal_metric',
                                                                                         ascending=False)
    return temporal_metric


def load_temporal_data():
    data = _parse_temporal_data()
    # last_period = '2022-Q1'
    total_malware = data[[f"n_malware.{yq}" for yq in year_quarters[:-2]]].iloc[0].sum()
    total_goodware = data[[f"n_goodware.{yq}" for yq in year_quarters[:-2]]].iloc[0].sum()
    total_temporal_samples = total_malware + total_goodware
    return data, total_temporal_samples


if __name__ == '__main__':
    print(compute_temporal_metric())