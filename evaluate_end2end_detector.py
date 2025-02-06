import argparse
import torch
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from maltorch.datasets.binary_dataset import BinaryDataset
from tqdm import tqdm
from maltorch.datasets.drs_dataset import DeRandomizedSmoothingDataset
from maltorch.datasets.random_drs_dataset import RandomDRSDataset
from maltorch.datasets.rs_dataset import RandomizedAblationDataset
from maltorch.datasets.rsdel_dataset import RandomizedDeletionDataset
from maltorch.datasets.sequential_drs_dataset import SequentialDRSDataset
from maltorch.zoo.model import BaseEmbeddingPytorchClassifier
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.ngramconv import NGramConv
from maltorch.zoo.shallowconv import ShallowConv
from secmlt.models.data_processing.data_processing import DataProcessing
from maltorch.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.data_processing.drs_preprocessing import DeRandomizedPreprocessing
from maltorch.data_processing.sequential_drs_preprocessing import SequentialDeRandomizedPreprocessing
from maltorch.data_processing.random_drs_preprocessing import RandomDeRandomizedPreprocessing
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing
from utils import read_json_file
from utils import check_cuda


device = check_cuda()

def get_preprocessing(configuration: dict) -> DataProcessing:
    try:
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
            return DeRandomizedPreprocessing(
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

def build_model(configuration: dict) -> tuple[BaseEmbeddingPytorchClassifier, DataProcessing, DataProcessing]:
    preprocessing = get_preprocessing(configuration)
    postprocessing = get_postprocessing(configuration)
    architecture_name = configuration["architecture"]
    if architecture_name == "MalConv":
        return MalConv.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            max_len=configuration["max_len"]
        ), preprocessing, postprocessing
    elif architecture_name == "AvastConv":
        return AvastStyleConv.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            max_len=configuration["max_len"]
        ), preprocessing, postprocessing
    elif architecture_name == "NGramConv":
        return NGramConv.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            max_len=configuration["max_len"]
        ), preprocessing, postprocessing
    elif architecture_name == "ShallowConv":
        return ShallowConv.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            max_len=configuration["max_len"]
        ), preprocessing, postprocessing
    elif architecture_name == "BBDnn":
        return BBDnn.create_model(
            model_path=configuration["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
        ), preprocessing, postprocessing
    else:
        raise ValueError(f"Model {architecture_name} not found")

def create_dataset(configuration: dict) -> Dataset:
    if configuration["dataset_type"] == "Binary":
        evaluation_dataset = BinaryDataset(
            csv_filepath=configuration["evaluation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"]
        )

    elif configuration["dataset_type"] == "RS":
        evaluation_dataset = RandomizedAblationDataset(
            csv_filepath=configuration["evaluation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            num_versions=configuration["num_versions"],
            pabl=configuration["pabl"],
            is_training=False
        )
    elif configuration["dataset_type"] == "RsDel":
        evaluation_dataset = RandomizedDeletionDataset(
            csv_filepath=configuration["evaluation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            num_versions=configuration["num_versions"],
            pdel=configuration["pdel"],
            is_training=False
        )
    elif configuration["dataset_type"] == "DRS":
        evaluation_dataset = DeRandomizedSmoothingDataset(
            csv_filepath=configuration["evaluation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            chunk_size=configuration["chunk_size"],
            is_training=False
        )
    elif configuration["dataset_type"] == "SequentialDRS":
        evaluation_dataset = SequentialDRSDataset(
            csv_filepath=configuration["evaluation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=False
        )
    elif configuration["dataset_type"] == "RandomDRS":
        evaluation_dataset = RandomDRSDataset(
            csv_filepath=configuration["evaluation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=False
        )
    else:
        raise ValueError(f"Dataset type {configuration['dataset_type']} not found. Please use one of the following: Binary, RS, RsDel, DRS, SequentialDRS, RandomDRS")
    return evaluation_dataset

def evaluate(model: BaseEmbeddingPytorchClassifier, dataloader: DataLoader)-> tuple[list[int], list[int]]:
    eval_total = 0
    eval_trues = []
    eval_preds = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            outputs = model.predict(x)
            outputs = outputs.squeeze()

            y_preds = outputs.round()
            eval_trues.append(y)
            eval_preds.append(y_preds)
            eval_total += y.size(0)
    return eval_preds, eval_trues

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
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate end2end malware detector')
    parser.add_argument("configuration_file",
                        type=str,
                        help="JSON-like file including the training and model configuration hyperparameters")
    args = parser.parse_args()

    configuration = read_json_file(args.configuration_file)
    model, preprocessing, postprocessing = build_model(configuration)

    dataset = create_dataset(configuration)
    num_workers = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)
    dataloader = DataLoader(
        dataset,
        batch_size=configuration["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.pad_collate_func)

    y_preds, y_trues = evaluate(model, dataloader)
    write_predictions(y_preds, y_trues, configuration["predictions_path"])
    write_metrics(y_preds, y_trues, configuration["metrics_path"])




