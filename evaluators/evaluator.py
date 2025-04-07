from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing
from maltorch.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.ngramconv import NGramConv
from maltorch.zoo.resnet18 import ResNet18
from utils import check_cuda, read_json_file
from pathlib import Path

# model_path = Path(__file__).parent.parent / 'maltorch' / 'models' / 'zoo' / 'MalConv'
#
# print(model_path)
#
# model = MalConv().create_model()


class Evaluator:
    def __init__(self, config_path):
        self.config = read_json_file(config_path)
        self.model = self.build_model(self.config)

    @staticmethod
    def build_model(config):
        architecture_name = config["architecture"]

        if architecture_name == "Malconv":
            return (
                MalConv().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models'  / architecture_name,
                )
            )
        if architecture_name == "AvastStyleConv":
            return (
                AvastStyleConv().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                )
            )
        if architecture_name == "BBDnn":
            return (
                BBDnn().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' /  architecture_name,
                )
            )
        if architecture_name == "NGramConv":
            return (
                NGramConv().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                )
            )
        if architecture_name == "ResNet18":
            return (
                ResNet18().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                    preprocessing=GrayscalePreprocessing(
                        width=256,
                        height=256,
                        convert_to_3d_image=True
                    )
                )
            )
        if architecture_name == "MalConvRS":
            return (
                MalConv().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                    preprocessing=RandomizedAblationPreprocessing(
                        pabl=0.20,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=MajorityVotingPostprocessing()
                )
            )
        if architecture_name == "AvastStyleConvRS":
            return (
                AvastStyleConv().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                    preprocessing=RandomizedAblationPreprocessing(
                        pabl=0.20,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=MajorityVotingPostprocessing()
                )
            )
        if architecture_name == "BBDnnRS":
            return (
                BBDnn().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                    preprocessing=RandomizedAblationPreprocessing(
                        pabl=0.20,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=MajorityVotingPostprocessing()
                )
            )
        if architecture_name == "NGramConvRS":
            return (
                NGramConv().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                    preprocessing=RandomizedAblationPreprocessing(
                        pabl=0.20,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=MajorityVotingPostprocessing()
                )
            )
        if architecture_name == "MalConvRsDel":
            return (
                MalConv().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                    preprocessing=RandomizedDeletionPreprocessing(
                        pdel=0.03,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=MajorityVotingPostprocessing()
                )
            )
        if architecture_name == "AvastStyleConvRsDel":
            return (
                AvastStyleConv().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                    preprocessing=RandomizedDeletionPreprocessing(
                        pdel=0.03,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=MajorityVotingPostprocessing()
                )
            )
        if architecture_name == "BBDnnRsDel":
            return (
                BBDnn().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                    preprocessing=RandomizedDeletionPreprocessing(
                        pdel=0.03,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=MajorityVotingPostprocessing()
                )
            )
        if architecture_name == "NGramConvRsDel":
            return (
                NGramConv().create_model(
                    model_path = Path(__file__).parent.parent / 'maltorch' / 'zoo' / 'models' / architecture_name,
                    preprocessing=RandomizedDeletionPreprocessing(
                        pdel=0.03,
                        num_versions=100,
                        padding_idx=256
                    ),
                    postprocessing=MajorityVotingPostprocessing()
                )
            )
        if architecture_name == "MalConvDRS":
            pass

    def evaluate(self, x):
        return self
