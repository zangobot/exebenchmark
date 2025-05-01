import argparse
import torch
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from maltorch.datasets.binary_dataset import BinaryDataset
from tqdm import tqdm
from maltorch.zoo.model import BaseEmbeddingPytorchClassifier, BasePytorchClassifier
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.ngramconv import NGramConv
from maltorch.zoo.shallowconv import ShallowConv
from maltorch.zoo.resnet18 import ResNet18
from secmlt.models.data_processing.data_processing import DataProcessing
from maltorch.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.data_processing.aisec_drs_preprocessing import DeRandomizedPreprocessing as AISECDeRandomizedPreprocessing
from maltorch.data_processing.sequential_drs_preprocessing import SequentialDeRandomizedPreprocessing
from maltorch.data_processing.random_drs_preprocessing import RandomDeRandomizedPreprocessing
from maltorch.data_processing.iclr_drs_preprocessing import DeRandomizedPreprocessing as ICLRDeRandomizedPreprocessing
from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing
from maltorch.data_processing.sigmoid_postprocessor import SigmoidPostprocessor
from utils import read_json_file, write_predictions, write_metrics, check_cuda


device = check_cuda()

def get_preprocessing(configuration: dict) -> DataProcessing:
    try:
        print("Preprocessing: ", configuration["preprocessing"])
        if configuration["preprocessing"] == "RS":
            return RandomizedAblationPreprocessing(
                pabl=configuration["pabl"],
                num_versions=configuration["num_versions"],
                padding_idx=configuration["padding_idx"]
            )
        elif configuration["preprocessing"] == "RsDel":
            return RandomizedDeletionPreprocessing(
                pdel=configuration["pdel"],
                num_versions=configuration["num_versions"],
                padding_idx=configuration["padding_idx"]
            )
        elif configuration["preprocessing"] == "DRS":
            return AISECDeRandomizedPreprocessing(
                chunk_size=configuration["chunk_size"],
                padding_idx=configuration["padding_idx"]
            )
        elif configuration["preprocessing"] == "SequentialDRS":
            return SequentialDeRandomizedPreprocessing(
                file_percentage=configuration["file_percentage"],
                num_chunks=configuration["num_chunks"],
                padding_idx=configuration["padding_idx"],
                min_chunk_size=configuration["min_chunk_size"]
            )
        elif configuration["preprocessing"] == "RandomDRS":
            return RandomDeRandomizedPreprocessing(
                file_percentage=configuration["file_percentage"],
                num_chunks=configuration["num_chunks"],
                padding_idx=configuration["padding_idx"],
                min_chunk_size=configuration["min_chunk_size"]
            )
        elif configuration["preprocessing"] == "DRSM":
            return ICLRDeRandomizedPreprocessing(
                num_chunks=configuration["num_chunks"],
                min_chunk_size=configuration["min_chunk_size"],
                padding_idx=configuration["padding_idx"]
            )
        elif configuration["preprocessing"] == "Grayscale":
            return GrayscalePreprocessing(
                width=configuration["width"],
                height=configuration["height"],
                convert_to_3d_image=configuration["convert_to_3d_image"]
            )
        else:
            return None
    except KeyError:
        print("Preprocessing: None")
        return None



def get_postprocessing(configuration: dict) -> DataProcessing:
    try:
        print("Postprocessing: ", configuration["postprocessing"])
        if configuration["postprocessing"] == "MajorityVoting":
            return MajorityVotingPostprocessing(apply_sigmoid=True)
        elif configuration["postprocessing"] == "Sigmoid":
            return SigmoidPostprocessor()
        else:
            raise ValueError(f"postprocessing {configuration['postprocessing']} not found")
    except KeyError:
        print("Postprocessing: None")
        return None

def build_model(configuration: dict) -> tuple[BasePytorchClassifier, DataProcessing, DataProcessing]:
    preprocessing = get_preprocessing(configuration)
    postprocessing = get_postprocessing(configuration)
    architecture_name = configuration["architecture"]
    print("Architecture: ", architecture_name)
    if architecture_name == "MalConv":
        return MalConv.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            kernel_size=configuration["kernel_size"] if "kernel_size" in configuration else None,
            stride=configuration["stride"] if "stride" in configuration else None,
        ), preprocessing, postprocessing
    elif architecture_name == "AvastConv":
        return AvastStyleConv.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
        ), preprocessing, postprocessing
    elif architecture_name == "NGramConv":
        return NGramConv.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
        ), preprocessing, postprocessing
    elif architecture_name == "ShallowConv":
        return ShallowConv.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
        ), preprocessing, postprocessing
    elif architecture_name == "BBDnn":
        return BBDnn.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
        ), preprocessing, postprocessing
    elif architecture_name == "ResNet18":
        return ResNet18.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"]
        ), preprocessing, postprocessing
    else:
        raise ValueError(f"classifier {architecture_name} not found")

def create_dataset(configuration: dict) -> Dataset:
    evaluation_dataset = BinaryDataset(
        csv_filepath=configuration["evaluation_file"],
        max_len=configuration["max_len"] if "max_len" in configuration else None,
        min_len=configuration["min_len"] if "min_len" in configuration else None,
        padding_idx=configuration["padding_idx"]
    )
    return evaluation_dataset

def evaluate(classifier: BasePytorchClassifier, dataloader: DataLoader)-> tuple[list[int], list[int]]:
    eval_trues = []
    eval_preds = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            y_preds = classifier.predict(x)
            eval_trues.append(y)
            eval_preds.append(y_preds)
    return eval_preds, eval_trues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate end2end malware detector')
    parser.add_argument("configuration_file",
                        type=str,
                        help="JSON-like file including the training and classifier configuration hyperparameters")
    args = parser.parse_args()

    configuration = read_json_file(args.configuration_file)
    classifier, preprocessing, postprocessing = build_model(configuration)

    dataset = create_dataset(configuration)
    num_workers = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)
    dataloader = DataLoader(
        dataset,
        batch_size=configuration["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.pad_collate_func)

    y_preds, y_trues = evaluate(classifier, dataloader)
    write_predictions(y_preds, y_trues, configuration["predictions_path"])
    write_metrics(y_preds, y_trues, configuration["metrics_path"])




