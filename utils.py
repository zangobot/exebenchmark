import json
import numpy as np
import csv
import torch

from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.model import Model
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.ngramconv import NGramConv
from maltorch.zoo.shallowconv import ShallowConv
from maltorch.zoo.resnet18 import ResNet18
from secmlt.models.data_processing.data_processing import DataProcessing
from maltorch.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing

from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing
from maltorch.data_processing.majority_voting_postprocessing import (
    MajorityVotingPostprocessing,
)


def read_json_file(filepath: str) -> dict:
    with open(filepath, "r") as input_file:
        return json.load(input_file)


def check_cuda():
    # Check if a GPU is available
    print("Is CUDA available?:", torch.cuda.is_available())
    # Check the number of GPUs
    print("Number of GPUs:", torch.cuda.device_count())
    # Check the name of the GPU
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_ember_csv(
    filepath: str, max_date: str = None, min_date: str = None
) -> np.array:
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    # Convert to numpy array (assuming all values are numerical)
    data = np.array(data, dtype=float)

    # Separate features and labels
    X, y = data[:, :-1], data[:, -1]

    return X, y


def write_predictions(y_preds, y_trues, predictions_file):
    with open(predictions_file, "w") as f:
        for y_pred, y_true in zip(y_preds, y_trues):
            f.write(f"{y_pred.item()},{y_true.item()}\n")


def write_metrics(y_preds, y_trues, metrics_file):
    """
    Calculate accuracy, precision, recall, and f1-score
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for y_pred, y_true in zip(y_preds, y_trues):
        if y_pred == 1 and y_true == 1:
            true_positives += 1
        elif y_pred == 1 and y_true == 0:
            false_positives += 1
        elif y_pred == 0 and y_true == 1:
            false_negatives += 1
        else:
            true_negatives += 1
    accuracy = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_positives + false_negatives
    )
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1-Score: {f1_score}\n")
        f.write("Confusion matrix: \n")
        f.write(f"[{true_negatives}, {false_positives}] \n")
        f.write(f"[{false_negatives}, {true_positives}] \n")


def get_preprocessing(configuration: dict) -> DataProcessing:
    try:
        if configuration["preprocessing"] == "RS":
            return RandomizedAblationPreprocessing(
                pabl=configuration["pabl"],
                num_versions=configuration["num_versions"],
                padding_idx=configuration["padding_idx"],
            )
        elif configuration["preprocessing"] == "RsDel":
            return RandomizedDeletionPreprocessing(
                pdel=configuration["pdel"],
                num_versions=configuration["num_versions"],
                padding_idx=configuration["padding_idx"],
            )
        elif configuration["preprocessing"] == "DRS":
            return DeRandomizedPreprocessing(
                chunk_size=configuration["chunk_size"],
                padding_idx=configuration["padding_idx"],
            )
        elif configuration["preprocessing"] == "SequentialDRS":
            return SequentialDeRandomizedPreprocessing(
                file_percentage=configuration["file_percentage"],
                num_chunks=configuration["num_chunks"],
                padding_idx=configuration["padding_idx"],
                min_chunk_size=configuration["min_chunk_size"],
            )
        elif configuration["preprocessing"] == "RandomDRS":
            return RandomDeRandomizedPreprocessing(
                file_percentage=configuration["file_percentage"],
                num_chunks=configuration["num_chunks"],
                padding_idx=configuration["padding_idx"],
                min_chunk_size=configuration["min_chunk_size"],
            )
        elif configuration["preprocessing"] == "Grayscale":
            return GrayscalePreprocessing(
                width=configuration["width"],
                height=configuration["height"],
                convert_to_3d_image=configuration["convert_to_3d_image"],
            )
        else:
            return None
    except KeyError:
        return None


def get_postprocessing(configuration: dict) -> DataProcessing:
    try:
        if configuration["postprocessing"] == "MajorityVoting":
            return MajorityVotingPostprocessing()
        else:
            return None
    except KeyError:
        return None


def build_model(
    configuration: dict,
) -> tuple[Model, DataProcessing, DataProcessing]:
    preprocessing = get_preprocessing(configuration)
    postprocessing = get_postprocessing(configuration)
    architecture_name = configuration["architecture"]
    if architecture_name == "EmberGBDT":
        return EmberGBDT.create_model(
            model_path=configuration["model_path"],
            device="cpu",
            preprocessing=None,
            postprocessing=None,
            trainer=None,
        )
    if architecture_name == "MalConv":
        return (
            MalConv.create_model(
                model_path=configuration["model_path"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                preprocessing=preprocessing,
                postprocessing=postprocessing,
                threshold=configuration["threshold"],
                padding_idx=configuration["padding_idx"],
                max_len=configuration["max_len"]
                if "max_len" in configuration
                else None,
            ),
            preprocessing,
            postprocessing,
        )
    elif architecture_name == "AvastConv":
        return (
            AvastStyleConv.create_model(
                model_path=configuration["model_path"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                preprocessing=preprocessing,
                postprocessing=postprocessing,
                threshold=configuration["threshold"],
                padding_idx=configuration["padding_idx"],
                max_len=configuration["max_len"]
                if "max_len" in configuration
                else None,
            ),
            preprocessing,
            postprocessing,
        )
    elif architecture_name == "NGramConv":
        return (
            NGramConv.create_model(
                model_path=configuration["model_path"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                preprocessing=preprocessing,
                postprocessing=postprocessing,
                threshold=configuration["threshold"],
                padding_idx=configuration["padding_idx"],
                max_len=configuration["max_len"]
                if "max_len" in configuration
                else None,
            ),
            preprocessing,
            postprocessing,
        )
    elif architecture_name == "ShallowConv":
        return (
            ShallowConv.create_model(
                model_path=configuration["model_path"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                preprocessing=preprocessing,
                postprocessing=postprocessing,
                threshold=configuration["threshold"],
                padding_idx=configuration["padding_idx"],
                max_len=configuration["max_len"]
                if "max_len" in configuration
                else None,
            ),
            preprocessing,
            postprocessing,
        )
    elif architecture_name == "BBDnn":
        return (
            BBDnn.create_model(
                model_path=configuration["model_path"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                preprocessing=preprocessing,
                postprocessing=postprocessing,
                threshold=configuration["threshold"],
                padding_idx=configuration["padding_idx"],
                max_len=configuration["max_len"]
                if "max_len" in configuration
                else None,
            ),
            preprocessing,
            postprocessing,
        )
    elif architecture_name == "ResNet18":
        return (
            ResNet18.create_model(
                model_path=configuration["model_path"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                preprocessing=preprocessing,
                postprocessing=postprocessing,
                threshold=configuration["threshold"],
            ),
            preprocessing,
            postprocessing,
        )
    else:
        raise ValueError(f"Model {architecture_name} not found")
