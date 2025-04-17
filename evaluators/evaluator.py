import abc

from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing
from maltorch.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.data_processing.sigmoid_postprocessor import SigmoidPostprocessor
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.ngramconv import NGramConv
from maltorch.zoo.resnet18 import ResNet18
from torch.utils.data import DataLoader

import utils
from config import ZOO_PATH
from utils import read_json_file


class Evaluator(abc.ABC):
    def __init__(self, config_path):
        self.config = read_json_file(config_path)
        self.model = self.build_model(self.config)

    @staticmethod
    def build_model(config):
        architecture_name = config["architecture"]
        device = utils.check_cuda()

        ablation_preprocessing = RandomizedAblationPreprocessing(pabl=0.20, num_versions=100, padding_idx=256)
        voting_postprocessing = MajorityVotingPostprocessing(apply_sigmoid=True)
        sigmoid_postprocessor = SigmoidPostprocessor()

        if architecture_name == "Malconv":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=device,
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
                    model_path=ZOO_PATH / architecture_name, device=device,
                    postprocessing=sigmoid_postprocessor
                )
            )
        if architecture_name == "BBDnn":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name, device=device,
                    postprocessing=sigmoid_postprocessor
                )
            )
        if architecture_name == "NGramConv":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name, device=device,
                    postprocessing=sigmoid_postprocessor
                )
            )
        if architecture_name == "ResNet18":
            return (
                ResNet18().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=device,
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
                    device=device,
                    preprocessing=ablation_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "AvastStyleConvRS":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=device,
                    preprocessing=ablation_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "BBDnnRS":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=device,
                    preprocessing=ablation_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "NGramConvRS":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=device,
                    preprocessing=ablation_preprocessing,
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "MalConvRsDel":
            return (
                MalConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=device,
                    preprocessing=RandomizedDeletionPreprocessing(
                        pdel=0.03,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "AvastStyleConvRsDel":
            return (
                AvastStyleConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=device,
                    preprocessing=RandomizedDeletionPreprocessing(
                        pdel=0.03,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "BBDnnRsDel":
            return (
                BBDnn().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=device,
                    preprocessing=RandomizedDeletionPreprocessing(
                        pdel=0.03,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "NGramConvRsDel":
            return (
                NGramConv().create_model(
                    model_path=ZOO_PATH / architecture_name,
                    device=device,
                    preprocessing=RandomizedDeletionPreprocessing(
                        pdel=0.03,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=voting_postprocessing
                )
            )
        if architecture_name == "MalConvDRS":
            pass

    @abc.abstractmethod
    def load_data(self) -> DataLoader:
        ...

    @abc.abstractmethod
    def evaluate(self, data_loader: DataLoader) -> DataLoader:
        ...
