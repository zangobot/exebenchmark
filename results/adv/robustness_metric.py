import numpy as np
from sklearn.metrics import auc
import pandas
from pathlib import Path

from results.constants import MODELS


def compute_robustness_metric():
    root = Path(__file__).parent
    data_folder = root / "data"
    all_adv_data = {m: {} for m in MODELS}
    all_eps_across_models = set()
    total_samples = 4764

    for m in MODELS:
        direct = data_folder / f"{m}_vs_{m}.csv"
        distance_per_hash = {}
        if direct.exists():
            data = pandas.read_csv(direct)
            data = data.replace(-1, np.inf)
            total_samples = len(data)
            for i, sample in data.iterrows():
                sample_hash = sample['hash']
                sample = sample.drop('hash')
                min_perturbation = sample.min()
                distance_per_hash[sample_hash] = min_perturbation
        for tr in data_folder.glob(f"*_vs_{m}.csv"):
            if tr == direct:
                continue
            tr_data = pandas.read_csv(tr).replace(-1, np.inf)
            for i, sample in tr_data.iterrows():
                sample_hash = sample['hash']
                perturbation_hash = sample.drop('hash')
                if sample_hash not in distance_per_hash:
                    distance_per_hash[sample_hash] = min(perturbation_hash)
                else:
                    distance_per_hash[sample_hash] = min(distance_per_hash[sample_hash], min(perturbation_hash))
        all_adv_data[m] = distance_per_hash
        for h in distance_per_hash:
            all_eps_across_models.add(distance_per_hash[h])

    all_eps_across_models = np.array(list(all_eps_across_models))
    all_eps_across_models.sort()

    robustness = {}

    for m in MODELS:
        adv_data = pandas.DataFrame(data=[[k, all_adv_data[m][k]] for k in all_adv_data[m]], columns=['hash', 'eps'])
        x = np.arange(len(all_eps_across_models) - 1)
        y = [total_samples - len(adv_data.loc[adv_data['eps'] <= eps]) for eps in all_eps_across_models[:-1]]
        auc_m = auc(x, y) / (total_samples * x[-1])
        robustness[m] = auc_m.item()
    robustness = pandas.DataFrame(data=[[m, robustness[m]] for m in robustness],
                                  columns=['model', 'robustness_metric']).sort_values(by='robustness_metric',
                                                                                      ascending=False)
    return robustness

if __name__ == '__main__':
    compute_robustness_metric()