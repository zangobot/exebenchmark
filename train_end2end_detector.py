import argparse
import torch
import json
from maltorch.datasets.binary_dataset import BinaryDataset
from maltorch.datasets.rs_dataset import RandomizedAblationDataset
from maltorch.datasets.rsdel_dataset import RandomizedDeletionDataset
from maltorch.datasets.random_drs_dataset import RandomDRSDataset
from maltorch.datasets.sequential_drs_dataset import SequentialDRSDataset
from maltorch.datasets.drs_dataset import DeRandomizedSmoothingDataset
from maltorch.trainers.early_stopping_pytorch_trainer import EarlyStoppingPyTorchTrainer
from torch.utils.data import DataLoader, Dataset
import multiprocessing
import os
from utils import read_json_file
from utils import check_cuda


device = check_cuda()

def build_model(configuration: dict) -> torch.nn.Module:
    architecture_name = configuration["architecture"]
    if architecture_name == "MalConv":
        from maltorch.zoo.malconv import MalConv
        return MalConv(
            embedding_size=configuration["embedding_size"],
            max_len=configuration["max_len"],
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"]
        )
    elif architecture_name == "AvastConv":
        from maltorch.zoo.avaststyleconv import AvastStyleConv
        return AvastStyleConv(
            embedding_size=configuration["embedding_size"],
            max_len=configuration["max_len"],
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"]
        )
    elif architecture_name == "NGramConv":
        from maltorch.zoo.ngramconv import NGramConv
        return NGramConv(
            embedding_size=configuration["embedding_size"],
            max_len=configuration["max_len"],
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"]
        )
    elif architecture_name == "ShallowConv":
        from maltorch.zoo.shallowconv import ShallowConv
        return ShallowConv(
            embedding_size=configuration["embedding_size"],
            max_len=configuration["max_len"],
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"]
        )
    elif architecture_name == "BBDnn":
        from maltorch.zoo.bbdnn import BBDnn
        return BBDnn() # Make sure to use padding_idx = 0 for the BBDnn model
    else:
        raise ValueError(f"Model {architecture_name} not found")


def create_datasets(configuration: dict) -> tuple[Dataset, Dataset]:
    if configuration["dataset_type"] == "binary":
        training_dataset = BinaryDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"]
        )
        validation_dataset = BinaryDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"]
        )
    elif configuration["dataset_type"] == "RS":
        training_dataset = RandomizedAblationDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            num_versions=configuration["num_versions"],
            pabl=configuration["pabl"],
            is_training=True
        )
        validation_dataset = RandomizedAblationDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            num_versions=configuration["num_versions"],
            pabl=configuration["pabl"],
            is_training=True
        )

    elif configuration["dataset_type"] == "RsDel":
        training_dataset = RandomizedDeletionDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            num_versions=configuration["num_versions"],
            pdel=configuration["pdel"],
            is_training=True
        )
        validation_dataset = RandomizedDeletionDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            num_versions=configuration["num_versions"],
            pdel=configuration["pdel"],
            is_training=True
        )

    elif configuration["dataset_type"] == "DRS":
        training_dataset = DeRandomizedSmoothingDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            chunk_size=configuration["chunk_size"],
            is_training=True
        )
        validation_dataset = DeRandomizedSmoothingDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            chunk_size=configuration["chunk_size"],
            is_training=True
        )
    elif configuration["dataset_type"] == "SequentialDRS":
        training_dataset = SequentialDRSDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True
        )
        validation_dataset = SequentialDRSDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True
        )
    elif configuration["dataset_type"] == "RandomDRS":
        training_dataset = RandomDRSDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True
        )
        validation_dataset = RandomDRSDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"],
            padding_idx=configuration["padding_idx"],
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True
        )

    else:
        raise ValueError(f"Dataset type {configuration['dataset_type']} not found. Please use one of the following: binary, RS, RsDel, DRS, SequentialDRS, RandomDRS")
    return training_dataset, validation_dataset


def save_results(trainer: EarlyStoppingPyTorchTrainer, configuration: dict):
    results = {
        "training_losses": trainer.training_losses,
        "training_accuracies": trainer.training_accuracies,
        "validation_losses": trainer.validation_losses,
        "validation_accuracies": trainer.validation_accuracies
    }
    with open(os.path.join(configuration["model_path"], "results.json"), "w") as output_file:
        json.dump(results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train end2end malware detector')
    parser.add_argument("configuration_file",
                        type=str,
                        help="JSON-like file including the training and model configuration hyperparameters")

    args = parser.parse_args()

    configuration = read_json_file(args.configuration_file)

    training_dataset, validation_dataset = create_datasets(configuration)

    num_workers = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)
    train_dataloader = DataLoader(
        training_dataset,
        batch_size=configuration["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=training_dataset.pad_collate_func)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=configuration["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=training_dataset.pad_collate_func
    )
    model = build_model(configuration)
    model = model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    trainer = EarlyStoppingPyTorchTrainer(
        optimizer,
        configuration["num_epochs"],
        criterion
    )
    model = trainer.train(
        model,
        train_dataloader,
        validation_dataloader,
        configuration["patience"]
    )
    if not os.path.exists(configuration["model_path"]):
        os.makedirs(configuration["model_path"])
    torch.save(model.state_dict(), os.path.join(configuration["model_path"],"model.pth"))
    save_results(trainer, configuration)





