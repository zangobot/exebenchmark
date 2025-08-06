import os
from multiprocessing import Pool
from pathlib import Path


from torch.utils.data import DataLoader

from evaluators.evaluator import Evaluator
from maltorch.adv.evasion.content_shift import ContentShift
from maltorch.adv.evasion.gamma_section_injection import GAMMASectionInjection
from config import BENIGNWARE_PATH, MALWARE_FOR_ADV
import torch
from maltorch.utils.utils import dump_torch_exe_to_file


from maltorch.datasets.binary_dataset import BinaryDataset

from maltorch.adv.evasion.base_optim_attack_creator import OptimizerBackends

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class AdversarialEvaluator(Evaluator):
    def __init__(self, config_path, device="cpu"):
        super().__init__(config_path, device=device)

        if "attack" not in self.config:
            raise ValueError("Attack configuration must contain an 'attack' key.")

        self.attack_engine = self.create_attack()
        self.examples_folder = (
            Path(self.config["examples_folder"])
            / self.config["architecture"]
            / self.config["attack"]
        )
        self.predictions_path = (
            Path(self.config["predictions_path"]) / self.config["architecture"]
        )
        if "transfer_path" in self.config and self.config["transfer_path"] is not None:
            self.transfer_path = (
            Path(self.config["transfer_path"])
            / self.config["architecture"]
            / self.config["attack"]
            )
        else:
            self.transfer_path = None

    def create_attack(self):
        if self.config["attack"] == "gamma":
            return GAMMASectionInjection(
                query_budget=500,
                benignware_folder=BENIGNWARE_PATH,
                which_sections=[".rdata"],
                how_many_sections=50,
                model_outputs_logits=False
            )

        if self.config["attack"] == "content_shift":
            return ContentShift(query_budget=500, perturbation_size=2048, 
                                model_outputs_logits=False,
                                backend=OptimizerBackends.NG)
        
    def _service_attack(self, dataloader, hashes, predictions_file):

        adv_dl = self.attack_engine(self.model, dataloader)
        print("Attacks completed, processing adversarial examples...")
        for idx, entry in enumerate(adv_dl.dataset):
            score = self.model(entry[0].unsqueeze(0))
            sample_hash = hashes[idx]
            dump_torch_exe_to_file(
                entry[0], str(self.examples_folder / f"{sample_hash}_adv")
            )
            with open(str(predictions_file), "a") as f:
                f.write(f"{sample_hash}_adv,{score.item()},1\n")
        

    def bulk_attack(self, n_jobs, batch_size=1):

        dataset = BinaryDataset(malware_directory=MALWARE_FOR_ADV)

        hashes = [f.name for f in Path(MALWARE_FOR_ADV).iterdir() if f.is_file()]
        indices = list(range(len(hashes)))
        chunks = [indices[i::n_jobs] for i in range(n_jobs)]

        self.examples_folder.mkdir(parents=True, exist_ok=True)
        self.predictions_path.mkdir(parents=True, exist_ok=True)

        predictions_file = self.predictions_path / f"{self.config['attack']}.csv"

        data_loaders = [
            DataLoader(
                torch.utils.data.Subset(dataset, chunk),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset.pad_collate_func,
            )
            for chunk in chunks
        ]
        if n_jobs == 1:
            adv_dl = self.attack_engine(self.model, data_loaders[0])
        else:
            with Pool(n_jobs) as pool:
                pool.starmap(
                    self._service_attack,
                    [(dl, [hashes[i] for i in chunk], predictions_file) for dl, chunk in zip(data_loaders, chunks)]
                )
            return 
                
        for idx, entry in enumerate(adv_dl.dataset):
            score = self.model(entry[0].unsqueeze(0))
            sample_hash = hashes[idx]
            dump_torch_exe_to_file(
                entry[0], str(self.examples_folder / f"{sample_hash}_adv")
            )
            with open(str(predictions_file), "a") as f:
                f.write(f"{sample_hash}_adv,{score.item()},1\n")

    def attacks_eval(self, examples_folder=None, predictions_file=None):
        predictions_path = (
            self.predictions_path
            if predictions_file is None
            else predictions_file.parent
        )
        examples_folder = (
            self.examples_folder if examples_folder is None else examples_folder
        )

        if predictions_file is None:
            self.predictions_path.mkdir(parents=True, exist_ok=True)
            predictions_file = self.predictions_path / f"{self.config['attack']}.csv"
        else:
            predictions_path.mkdir(parents=True, exist_ok=True)

        dataset = BinaryDataset(malware_directory=examples_folder)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with open(str(predictions_file), "a") as f:
            with torch.no_grad():
                for batch in data_loader:
                    x = batch[0]
                    x = x.to(self.device)
                    pred = self.model(x)
                    pred = pred.cpu().numpy()
                    for i in range(len(pred)):
                        f.write(f"{pred[i][0]},{1}\n")

    def transfer_eval(self):
        
        # This evaluates all adversarial examples for all models for the attack specified in the config
        base_path = Path(self.config["examples_folder"])
        for subdir in base_path.iterdir():
            if subdir.is_dir() and subdir.name != self.config["architecture"]:
                target = subdir / self.config["attack"]
                if target.exists():
                    output = self.transfer_path / f"{subdir.name}.csv"
                    self.attacks_eval(target, output)
