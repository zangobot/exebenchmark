import os
from multiprocessing import Pool
from pathlib import Path

import lief

from torch.utils.data import DataLoader, TensorDataset

from evaluators.evaluator import Evaluator
from maltorch.adv.evasion.content_shift import ContentShift
from maltorch.adv.evasion.gamma_section_injection import GAMMASectionInjection
from config import BENIGNWARE_PATH, MALWARE_FOR_ADV
from maltorch.adv.evasion.padding import Padding
from secmlt.metrics.classification import Accuracy
import torch
from maltorch.utils.utils import dump_torch_exe_to_file

from maltorch.data.loader import load_from_folder, create_labels

from config import MALWARE_FOR_ADV, BENIGNWARE_PATH


class AdversarialEvaluator(Evaluator):

    def __init__(self, config_path, device="cpu"):
        super().__init__(config_path, device=device)

        if "attack" not in self.config:
            raise ValueError("Attack configuration must contain an 'attack' key.")

        self.attack_engine = self.create_attack()
        self.examples_folder = Path(self.config["examples_folder"]) / self.config["architecture"] / self.config["attack"]
        self.predictions_path = Path(self.config["predictions_path"]) / self.config["architecture"]
        self.transfer_path = Path(self.config["transfer_path"]) / self.config["architecture"] / self.config["attack"]


    def create_attack(self):

        if self.config["attack"] == "gamma":
            return GAMMASectionInjection(
                query_budget=20,
                benignware_folder=BENIGNWARE_PATH,
                which_sections=[".rdata"],
                how_many_sections=50
            )

        if self.config["attack"] == "content_shift":
            return ContentShift(
                query_budget=500,
                perturbation_size=2048
            )

    def bulk_attack(self, n_jobs = 1, batch_size = 2):
        X = load_from_folder(MALWARE_FOR_ADV, limit=4)
        y = create_labels(X, 1)
        hashes = [f.name for f in Path(MALWARE_FOR_ADV).iterdir() if f.is_file()][:4]
        indices = list(range(len(X)))
        chunks = [indices[i::n_jobs] for i in range(n_jobs)]

        data_loaders = [
            DataLoader(TensorDataset(X[chunk], y[chunk]), batch_size=batch_size, shuffle=False)
            for chunk in chunks
        ]

        if n_jobs == 1:
            adv_dl = self.attack_engine(self.model, data_loaders[0])
        else:
            with Pool(n_jobs) as pool:
                adv_dl = pool.starmap(self.attack_engine, [(self.model, dl) for dl in data_loaders])
                all_datasets = [dl.dataset for dl in adv_dl]
                merged_dataset = torch.utils.data.ConcatDataset(all_datasets)
                adv_dl = DataLoader(merged_dataset, batch_size=batch_size, shuffle=False)

        self.examples_folder.mkdir(parents=True, exist_ok=True)
        self.predictions_path.mkdir(parents=True, exist_ok=True)

        predictions_file = self.predictions_path / f"{self.config['attack']}.csv"

        for idx, entry in enumerate(adv_dl.dataset):
            score = self.model(entry[0].unsqueeze(0))
            sample_hash = hashes[idx]
            if score < 1:
                dump_torch_exe_to_file(entry[0], str(self.examples_folder / f"{sample_hash}_adv"))
                with open(str(predictions_file), "a") as f:
                    f.write(f"{sample_hash}_adv,{score.item()},1\n")


    def attacks_eval(self, examples_folder=None, predictions_file=None):

        predictions_path = self.predictions_path if predictions_file is None else predictions_file.parent
        examples_folder = self.examples_folder if examples_folder is None else examples_folder

        if predictions_file is None:
            self.predictions_path.mkdir(parents=True, exist_ok=True)
            predictions_file = self.predictions_path / f"{self.config['attack']}.csv"
        else:
            predictions_path.mkdir(parents=True, exist_ok=True)


        X = load_from_folder(examples_folder)
        data_loader = DataLoader(TensorDataset(X), batch_size=1, shuffle=False)
        with open(str(predictions_file), "a") as f:
            with torch.no_grad():
                for batch in data_loader:
                    x = batch[0]
                    x = x.to(self.device)
                    pred = self.model(x)
                    pred = pred.cpu().numpy()
                    if pred[0] < self.model.threshold:
                        y = 1
                    else:
                        y = 0
                    for i in range(len(pred)):
                        f.write(f"{pred[i][0]},{y}\n")

    def transfer_eval(self):
        # This evaluates all adversarial examples for all models for the attack specified in the config
        base_path = Path(self.config["examples_folder"])
        for subdir in base_path.iterdir():
            if subdir.is_dir() and subdir.name != self.config["architecture"]:
                target = subdir / self.config["attack"]
                if target.exists():
                    output = self.transfer_path / f"{subdir.name}.csv"
                    self.attacks_eval(target, output)


