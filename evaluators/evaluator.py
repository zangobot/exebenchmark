import abc

import torch

import maltorch.data.loader
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

import utils
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
        fsc_drs_preprocessing = FixedSizeChunkDeRandomizedPreprocessing(chunk_size=512, padding_idx=256) #check
        kp_drs_preprocessing = KPartitionDeRandomizedPreprocessing(num_chunks=4, min_chunk_size=500, padding_idx=256) #check
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
        if architecture_name == "MalConvDRS":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "AvastStyleConvDRS":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "BBDnnDRS":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "NGramConvDRS":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=drs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "MalConvRDRS":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=rdrs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "AvastStyleConvRDRS":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=rdrs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "BBDnnRDRS":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=rdrs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "NGramConvRDRS":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=self.device,
                    preprocessing=rdrs_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        raise NotImplementedError(f"Model {architecture_name} not implemented.")

    def load_data(self) -> DataLoader:
        max_date = self.config.get("max_date")
        min_date = self.config.get("min_date")

        if not max_date or max_date.lower() == "none":
            max_date = None
        if not min_date or min_date.lower() == "none":
            min_date = None

        metadata_path = self.config["metadata_path"]

        if max_date is None and min_date is None:
            dataset = BinaryDataset(
                csv_filepath=metadata_path
            )
            return DataLoader(
                dataset,
                shuffle=False,
                batch_size=1
            )
        else:
            dataset = BinaryDataset(
                csv_filepath=metadata_path,
                max_date=max_date,
                min_date=min_date
            )
            return DataLoader(
                dataset,
                shuffle=False,
                batch_size=1,
            )

    def evaluate(self, output_path: str) -> None:

        data_loader = self.load_data()

        with open(output_path, "a") as f:
            with torch.no_grad():
                for batch in data_loader:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.model(x)
                    pred = pred.cpu().numpy()
                    y = y.cpu().numpy()
                    for i in range(len(pred)):
                        f.write(f"{pred[i]},{y[i]}\n")



