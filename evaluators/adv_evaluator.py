from multiprocessing import Pool
from pathlib import Path

import lief
from torch.utils.data import DataLoader, TensorDataset

from evaluators.evaluator import Evaluator
from maltorch.adv.evasion.gamma_section_injection import GAMMASectionInjection
from config import BENIGNWARE_PATH, MALWARE_FOR_ADV
from maltorch.adv.evasion.padding import Padding
from secmlt.metrics.classification import Accuracy
import torch


def load_from_folder(
    path: Path, extension: str = "exe", padding: int = 256, limit=None, device="cpu"
) -> torch.Tensor:
    """Create a torch.Tensor whose rows are all the file with extension specified in input.
    Tensor are padded to match the same size.
    :param path: Folder path
    :param extension: default "exe", filters all the file based on this extension
    :param padding: default 256, pad every tensor with this value to uniform the size
    :param limit: default None, limit the number of loaded file, None for load all folder
    :return: a torch.Tensor containing all the file converted into tensors
    """
    X = []
    for filepath in path.glob(f"*"):
        if lief.is_pe(str(filepath)):
            x = load_single_exe(filepath)
            X.append(x)
            if limit is not None and len(X) >= limit:
                break
        else:
            continue
    X = torch.nn.utils.rnn.pad_sequence(X, padding_value=padding).transpose(0, 1).long()
    X = X.to(device)
    return X


def create_labels(x: torch.Tensor, label: int, device="cpu"):
    """
    Create the labels for the specified data.
    """
    y = torch.zeros((x.shape[0], 1)) + label
    y = y.to(device)
    return y


def load_single_exe(path: Path) -> torch.Tensor:
    """
    Create a torch.Tensor from the file pointed in the path
    :param path: a pathlib Path
    :return: torch.Tensor containing the bytes of the file as a tensor
    """
    with open(path, "rb") as h:
        code = h.read()
    x = torch.frombuffer(bytearray(code), dtype=torch.uint8).to(torch.float)
    return x




class AdversarialEvaluator(Evaluator):

    def __init__(self, config_path):
        super().__init__(config_path)

        if "attack" not in self.config:
            raise ValueError("Attack configuration must contain an 'attack' key.")

        self.attack_engine = self.create_attack()

    def create_attack(self):

        if self.config["attack"] == "gamma":
            return GAMMASectionInjection(
                query_budget=500,
                benignware_folder=Path("/Users/bridge/PhD/Code/obelisk/data/win_exe/win11/syswow64/"),
                which_sections=[".rdata"],
                how_many_sections=50
            )

        if self.config["attack"] == "padding":
            return Padding(
                query_budget=500,
                padding=4096
            )


    def bulk_attack(self, n_jobs = 1, batch_size = 1):
        X = load_from_folder(Path("/Users/bridge/PhD/Code/mal-pipeline/malware/"), "", limit=2)
        y = create_labels(X, 1)

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

        # print("Accuracy: ", Accuracy()(self.model, adv_dl))
        for entry in adv_dl.dataset:
            print(self.model(entry[0].unsqueeze(0)))
            print(entry)



