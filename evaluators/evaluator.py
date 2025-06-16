import abc
import os
import sys

import torch

from maltorch.data_processing.dynamic_sequential_drs_preprocessing import (
    DynamicSequentialDeRandomizedPreprocessing,
)
from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing
from maltorch.data_processing.majority_voting_postprocessing import (
    MajorityVotingPostprocessing,
)
from maltorch.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.data_processing.dynamic_random_drs_preprocessing import (
    DynamicRandomDeRandomizedPreprocessing,
)
from maltorch.data_processing.fixed_size_chunk_drs_preprocessing import (
    FixedSizeChunkDeRandomizedPreprocessing,
)
from maltorch.data_processing.k_partition_drs_preprocessing import (
    KPartitionDeRandomizedPreprocessing,
)
from maltorch.data_processing.sigmoid_postprocessor import SigmoidPostprocessor
from maltorch.datasets.binary_dataset import BinaryDataset
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.ngramconv import NGramConv
from maltorch.zoo.resnet18 import ResNet18
from torch.utils.data import DataLoader
from pathlib import Path
import lightgbm as lgb


# all models should be downloaded in ZOO_PATH folder of exebenchmark
from config import ZOO_PATH
from utils import read_json_file
from utils import load_ember_csv
from typing import Union
import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, config: Union[str, dict] = None, device: str = None):
        if isinstance(config, str):
            self.config = read_json_file(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError("config must be a str (path) or a dict.")
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.build_model(self.config)

    def build_model(self, config):
        architecture_name = config["architecture"]

        ablation_preprocessing = RandomizedAblationPreprocessing(
            pabl=0.10, num_versions=100, padding_idx=256
        )
        deletion_preprocessing = RandomizedDeletionPreprocessing(
            pdel=0.10, num_versions=100, padding_idx=256
        )
        dynamic_rdrs_preprocessing = DynamicRandomDeRandomizedPreprocessing(
            file_percentage=0.10, num_chunks=100, padding_idx=256
        )  # min chunk to be defined model dependently
        dynamic_sdrs_preprocessing = DynamicSequentialDeRandomizedPreprocessing(
            file_percentage=0.10, num_chunks=100, padding_idx=256
        )
        f_drs_preprocessing = FixedSizeChunkDeRandomizedPreprocessing(
            chunk_size=32768, padding_idx=256
        )
        k_drs_preprocessing = KPartitionDeRandomizedPreprocessing(
            num_chunks=12, padding_idx=256
        )  # min chunk to be defined model dependently
        voting_postprocessing = MajorityVotingPostprocessing(apply_sigmoid=True)
        sigmoid_postprocessor = SigmoidPostprocessor()

        if architecture_name == "MalConv":
            return MalConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                postprocessing=sigmoid_postprocessor,
            )
        if architecture_name == "EmberGBDT":
            return EmberGBDT().create_model(
                model_path=ZOO_PATH / architecture_name, device="cpu"
            )
        if architecture_name == "AvastStyleConv":
            return AvastStyleConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                postprocessing=sigmoid_postprocessor,
            )
        if architecture_name == "BBDnn":
            return BBDnn().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                postprocessing=sigmoid_postprocessor,
            )
        if architecture_name == "NGramConv":
            return NGramConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                postprocessing=sigmoid_postprocessor,
            )
        if architecture_name == "ResNet18":
            return ResNet18().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=GrayscalePreprocessing(
                    width=256,
                    height=256,
                    convert_to_3d_image=True,
                ),
                postprocessing=sigmoid_postprocessor,
            )
        if architecture_name == "MalConvRS":
            return MalConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=ablation_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "AvastStyleConvRS":
            return AvastStyleConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=ablation_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "BBDnnRS":
            return BBDnn().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=ablation_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "NGramConvRS":
            return NGramConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=ablation_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "MalConvRsDel":
            return MalConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=deletion_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "AvastStyleConvRsDel":
            return AvastStyleConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=deletion_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "BBDnnRsDel":
            return BBDnn().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=deletion_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "NGramConvRsDel":
            return NGramConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=deletion_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "MalConvRDRS":
            dynamic_rdrs_preprocessing.min_chunk_size = 500
            return MalConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=dynamic_rdrs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "AvastStyleConvRDRS":
            dynamic_rdrs_preprocessing.min_chunk_size = 102400
            return AvastStyleConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=dynamic_rdrs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "BBDnnRDRS":
            dynamic_rdrs_preprocessing.min_chunk_size = 4096
            return BBDnn().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=dynamic_rdrs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "NGramConvRDRS":
            dynamic_rdrs_preprocessing.min_chunk_size = 512
            return NGramConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=dynamic_rdrs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "MalConvSDRS":
            dynamic_sdrs_preprocessing.min_chunk_size = 500
            return MalConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=dynamic_sdrs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "AvastStyleConvSDRS":
            dynamic_sdrs_preprocessing.min_chunk_size = 102400
            return AvastStyleConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=dynamic_sdrs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "BBDnnSDRS":
            dynamic_sdrs_preprocessing.min_chunk_size = 4096
            return BBDnn().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=dynamic_sdrs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "NGramConvSDRS":
            dynamic_sdrs_preprocessing.min_chunk_size = 512
            return NGramConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=dynamic_sdrs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "MalConvFDRS":
            return MalConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=f_drs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "AvastStyleConvFDRS":
            return AvastStyleConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=f_drs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "BBDnnFDRS":
            return BBDnn().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=f_drs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "NGramConvFDRS":
            return NGramConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=f_drs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "MalConvKDRS":
            k_drs_preprocessing.min_chunk_size = 500
            return MalConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=k_drs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "AvastStyleConvKDRS":
            k_drs_preprocessing.min_chunk_size = 102400
            return AvastStyleConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=k_drs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "BBDnnKDRS":
            k_drs_preprocessing.min_chunk_size = 4096
            return BBDnn().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=k_drs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        if architecture_name == "NGramConvKDRS":
            k_drs_preprocessing.min_chunk_size = 512
            return NGramConv().create_model(
                model_path=ZOO_PATH / architecture_name,
                device=self.device,
                preprocessing=k_drs_preprocessing,
                postprocessing=voting_postprocessing,
            )
        raise NotImplementedError(f"Model {architecture_name} not implemented.")

    def load_data(self, batch_size=None) -> Union[DataLoader, np.array]:
        # Set batch_size=1 only if model preprocessing is set and is not one of the known preprocessing instances
        known_preprocessings = (
            RandomizedAblationPreprocessing,
            RandomizedDeletionPreprocessing,
            DynamicRandomDeRandomizedPreprocessing,
            FixedSizeChunkDeRandomizedPreprocessing,
            KPartitionDeRandomizedPreprocessing,
            GrayscalePreprocessing,
        )
        if self.model._preprocessing is not None and isinstance(
            self.model._preprocessing, known_preprocessings
        ):
            batch_size = 1

        max_date = self.config["max_date"]
        min_date = self.config["min_date"]

        if not max_date or max_date.lower() == "none":
            max_date = None
        if not min_date or min_date.lower() == "none":
            min_date = None

        metadata_path = self.config["metadata_path"]

        if max_date is None and min_date is None:
            if self.config["architecture"] == "EmberGBDT":
                dataset = BinaryDataset(
                    csv_filepath=metadata_path,
                )
                return DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=1,
                )
            else:
                dataset = BinaryDataset(
                    csv_filepath=metadata_path
                )
                return DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=batch_size,
                    collate_fn=dataset.pad_collate_func,
                )

        else:
            if self.config["architecture"] == "EmberGBDT":
                dataset = BinaryDataset(
                    csv_filepath=metadata_path,
                    max_date=max_date,
                    min_date=min_date,
                )
                return DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=1,
                )

            else:
                dataset = BinaryDataset(
                    csv_filepath=metadata_path,
                    max_date=max_date,
                    min_date=min_date,
                )
                return DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=batch_size,
                    collate_fn=dataset.pad_collate_func,
                )

    def evaluate(self, batch_size: int = None) -> None:
        predictions_folder = self.config.get("predictions_folder")
        predictions_path = (
            Path(predictions_folder) / f"{self.config['architecture']}.csv"
        )
        Path(predictions_folder).mkdir(parents=True, exist_ok=True)

        if self.config["architecture"] == "EmberGBDT":
            data_loader = self.load_data()
            with open(predictions_path, "a") as f:
                for batch in data_loader:
                    x, y = batch
                    x = x.to(torch.device("cpu"))
                    y = y.to(torch.device("cpu"))
                    pred = self.model(x)
                    pred = pred.numpy()
                    y = y.numpy()
                    for i in range(len(pred)):
                        f.write(f"{pred[i][0]},{y[i]}\n")

        else:
            data_loader = self.load_data(batch_size)

            if predictions_folder is None:
                raise ValueError(
                    "Output path for predictions is not specified in the config."
                )

            with open(predictions_path, "a") as f:
                with torch.no_grad():
                    for batch in data_loader:
                        x, y = batch
                        x = x.to(self.device)
                        y = y.to(self.device)
                        pred = self.model(x)
                        pred = pred.cpu().numpy()
                        y = y.cpu().numpy()
                        for i in range(len(pred)):
                            f.write(f"{pred[i][0]},{y[i]}\n")
