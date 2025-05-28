import abc
import os 
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'maltorch', 'src')))

import torch
from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing
from maltorch.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.data_processing.dynamic_random_drs_preprocessing import DynamicRandomDeRandomizedPreprocessing
from maltorch.data_processing.fixed_size_chunk_drs_preprocessing import FixedSizeChunkDeRandomizedPreprocessing
from maltorch.data_processing.k_partition_drs_preprocessing import KPartitionDeRandomizedPreprocessing
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

# import utils
# all models should be downloaded in ZOO_PATH folder of exebenchmark
from config import ZOO_PATH
from utils import read_json_file


class Evaluator:
    def __init__(self, config_path):
        self.config = read_json_file(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.build_model(self.config)


    def build_model(self, config):
        architecture_name = config["architecture"]

        ablation_preprocessing = RandomizedAblationPreprocessing(pabl=0.20, num_versions=100, padding_idx=256)
        deletion_preprocessing = RandomizedDeletionPreprocessing(pdel=0.03, num_versions=100, padding_idx=256)
        dynamic_rdrs_preprocessing = DynamicRandomDeRandomizedPreprocessing(file_percentage=0.05, num_chunks=100, padding_idx=256, min_chunk_size=500) #check
        f_drs_preprocessing = FixedSizeChunkDeRandomizedPreprocessing(chunk_size=512, padding_idx=256) #check
        k_drs_preprocessing = KPartitionDeRandomizedPreprocessing(num_chunks=4, min_chunk_size=500, padding_idx=256) #check
        voting_postprocessing = MajorityVotingPostprocessing(apply_sigmoid=True)
        sigmoid_postprocessor = SigmoidPostprocessor()

        if architecture_name == "MalConv":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    postprocessing=sigmoid_postprocessor
                )
            )
        if architecture_name == "EmberGBDT":
            return (
                EmberGBDT().create_model(
                    model_path=ZOO_PATH / architecture_name
                )
            )
        if architecture_name == "AvastStyleConv":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name, device=self.device,
                    postprocessing=sigmoid_postprocessor
                )
            )
        if architecture_name == "BBDnn":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name, device=self.device,
                    postprocessing=sigmoid_postprocessor
                )
            )
        if architecture_name == "NGramConv":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name, device=self.device,
                    postprocessing=sigmoid_postprocessor
                )
            )
        if architecture_name == "ResNet18":
            return (
                ResNet18().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=GrayscalePreprocessing(
                        width=256,
                        height=256,
                        convert_to_3d_image=True,
                    )
                )
            )

        if architecture_name == "MalConvRS":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=ablation_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "AvastStyleConvRS":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=ablation_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "BBDnnRS":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=ablation_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "NGramConvRS":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=ablation_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "MalConvRsDel":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=deletion_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "AvastStyleConvRsDel":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=deletion_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "BBDnnRsDel":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=deletion_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "NGramConvRsDel":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=deletion_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "MalConvRDRS":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=dynamic_rdrs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "AvastStyleConvRDRS":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=dynamic_rdrs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "BBDnnRDRS":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=dynamic_rdrs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "NGramConvRDRS":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=dynamic_rdrs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "MalConvFDRS":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=f_drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "AvastStyleConvFDRS":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=f_drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "BBDnnFDRS":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=f_drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "NGramConvFDRS":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=f_drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "MalConvKDRS":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=k_drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "AvastStyleConvKDRS":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=k_drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "BBDnnKDRS":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=k_drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "NGramConvKDRS":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=k_drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        raise NotImplementedError(f"Model {architecture_name} not implemented.")

    def load_data(self, batch_size: int = 32) -> DataLoader:
        max_date = self.config["max_date"]
        min_date = self.config["min_date"]

        if not max_date or max_date.lower() == "none":
            max_date = None
        if not min_date or min_date.lower() == "none":
            min_date = None

        metadata_path = self.config["metadata_path"]

        if max_date is None and min_date is None:
            dataset = BinaryDataset(
                csv_filepath=metadata_path, 
                max_len=self.model.model.max_len if hasattr(self.model.model, 'max_len') else None,
                min_len=self.model.model.min_len if hasattr(self.model.model, 'min_len') else None
            )
            return DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size, 
                collate_fn=dataset.pad_collate_func
            )
        else:
            dataset = BinaryDataset(
                csv_filepath=metadata_path,
                max_date=max_date,
                min_date=min_date,
                max_len=self.model.model.max_len if hasattr(self.model.model, 'max_len') else None,
                min_len=self.model.model.min_len if hasattr(self.model.model, 'min_len') else None
            )
            return DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                collate_fn=dataset.pad_collate_func
            )

    def evaluate(self, batch_size: int = 32) -> None:

        data_loader = self.load_data(batch_size)

        predictions_folder = self.config.get("predictions_folder")

        if predictions_folder is None:
            raise ValueError("Output path for predictions is not specified in the config.")

        predictions_path = Path(predictions_folder) / f"{self.config['architecture']}.csv"

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

    



