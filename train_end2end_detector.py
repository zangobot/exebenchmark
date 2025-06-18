import argparse
import torch
import json
from maltorch.datasets.binary_dataset import BinaryDataset
from maltorch.datasets.rs_dataset import RandomizedAblationDataset
from maltorch.datasets.rsdel_dataset import RandomizedDeletionDataset
from maltorch.datasets.dynamic_random_drs_dataset import DynamicRandomDRSDataset
from maltorch.datasets.dynamic_sequential_drs_dataset import DynamicSequentialDRSDataset
from maltorch.datasets.fixed_size_chunk_drs_dataset import FixedSizeChunkDeRandomizedSmoothingDataset
from maltorch.datasets.k_partition_drs_dataset import KPartitionDeRandomizedSmoothingDataset
from maltorch.datasets.grayscale_dataset import GrayscaleDataset
from maltorch.datasets.random_chunk_sampler import RandomChunkSampler
from maltorch.trainers.early_stopping_pytorch_trainer import EarlyStoppingPyTorchTrainer
from torch.utils.data import DataLoader, Dataset
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.ngramconv import NGramConv
from maltorch.zoo.shallowconv import ShallowConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.resnet18 import ResNet18
import multiprocessing
import os
from utils import read_json_file
from utils import check_cuda


device = check_cuda()


def build_model(configuration: dict) -> torch.nn.Module:
    architecture_name = configuration["architecture"]
    if architecture_name == "MalConv":
        return MalConv(
            embedding_size=configuration["embedding_size"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
            kernel_size=configuration["kernel_size"] if "kernel_size" in configuration else None,
            stride=configuration["stride"] if "stride" in configuration else None,
        )
    elif architecture_name == "AvastConv":
        return AvastStyleConv(
            embedding_size=configuration["embedding_size"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
        )
    elif architecture_name == "NGramConv":
        return NGramConv(
            embedding_size=configuration["embedding_size"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
        )
    elif architecture_name == "ShallowConv":
        return ShallowConv(
            embedding_size=configuration["embedding_size"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
        )
    elif architecture_name == "BBDnn":
        return BBDnn(
            embedding_size=configuration["embedding_size"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            threshold=configuration["threshold"],
            padding_idx=configuration["padding_idx"],
        )
    elif architecture_name == "ResNet18":
        return ResNet18()
    else:
        raise ValueError(f"Model {architecture_name} not found")


def create_datasets(
    configuration: dict,
) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
    num_workers = max(
        multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1
    )
    if configuration["dataset_type"] == "Binary":
        training_dataset = BinaryDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
        )
        validation_dataset = BinaryDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
        )
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=configuration["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=training_dataset.pad_collate_func,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=validation_dataset.pad_collate_func
        )
    elif configuration["dataset_type"] == "RS":
        training_dataset = RandomizedAblationDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            num_versions=configuration["num_versions"],
            pabl=configuration["pabl"],
            is_training=True,
        )
        validation_dataset = RandomizedAblationDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            num_versions=configuration["num_versions"],
            pabl=configuration["pabl"],
            is_training=True,
        )
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=configuration["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=training_dataset.pad_collate_func,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=validation_dataset.pad_collate_func
        )
    elif configuration["dataset_type"] == "RsDel":
        training_dataset = RandomizedDeletionDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            num_versions=configuration["num_versions"],
            pdel=configuration["pdel"],
            is_training=True,
        )
        validation_dataset = RandomizedDeletionDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            num_versions=configuration["num_versions"],
            pdel=configuration["pdel"],
            is_training=True,
        )
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=configuration["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=training_dataset.pad_collate_func,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=validation_dataset.pad_collate_func
        )
    elif configuration["dataset_type"] == "F-DRS":
        training_dataset = FixedSizeChunkDeRandomizedSmoothingDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            chunk_size=configuration["chunk_size"],
            is_training=True,
        )
        validation_dataset = FixedSizeChunkDeRandomizedSmoothingDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            chunk_size=configuration["chunk_size"],
            is_training=True,
        )
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=configuration["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=training_dataset.pad_collate_func,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=validation_dataset.pad_collate_func
        )
    elif configuration["dataset_type"] == "SequentialDRS":
        training_dataset = DynamicSequentialDRSDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            sort_by_size=configuration["sort_by_size"]
            if "sort_by_size" in configuration
            else None,
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True,
        )
        validation_dataset = DynamicSequentialDRSDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            sort_by_size=configuration["sort_by_size"]
            if "sort_by_size" in configuration
            else None,
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True,
        )
        if configuration["sort_by_size"] is True:
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=configuration["batch_size"],
                num_workers=num_workers,
                collate_fn=training_dataset.pad_collate_func,
                sampler=RandomChunkSampler(
                    training_dataset, configuration["batch_size"]
                ),
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=1,
                num_workers=num_workers,
                collate_fn=training_dataset.pad_collate_func,
                sampler=RandomChunkSampler(validation_dataset, 1)
            )
        else:
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=configuration["batch_size"],
                shuffle=True,
                num_workers=num_workers,
                collate_fn=training_dataset.pad_collate_func)
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=validation_dataset.pad_collate_func
            )
    elif configuration["dataset_type"] == "RandomDRS":
        training_dataset = DynamicRandomDRSDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            sort_by_size=configuration["sort_by_size"]
            if "sort_by_size" in configuration
            else None,
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True,
        )
        validation_dataset = DynamicRandomDRSDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            sort_by_size=configuration["sort_by_size"]
            if "sort_by_size" in configuration
            else None,
            file_percentage=configuration["file_percentage"],
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True,
        )
        if configuration["sort_by_size"] is True:
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=configuration["batch_size"],
                num_workers=num_workers,
                collate_fn=training_dataset.pad_collate_func,
                sampler=RandomChunkSampler(
                    training_dataset, configuration["batch_size"]
                ),
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=1,
                num_workers=num_workers,
                collate_fn=validation_dataset.pad_collate_func,
                sampler=RandomChunkSampler(validation_dataset, configuration["batch_size"])
            )
        else:
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=configuration["batch_size"],
                shuffle=True,
                num_workers=num_workers,
                collate_fn=training_dataset.pad_collate_func,
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=validation_dataset.pad_collate_func
            )
    elif configuration["dataset_type"] == "K-DRS":
        training_dataset = KPartitionDeRandomizedSmoothingDataset(
            csv_filepath=configuration["training_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            sort_by_size=configuration["sort_by_size"] if "sort_by_size" in configuration else None,
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True
        )
        validation_dataset = KPartitionDeRandomizedSmoothingDataset(
            csv_filepath=configuration["validation_file"],
            max_len=configuration["max_len"] if "max_len" in configuration else None,
            padding_idx=configuration["padding_idx"],
            min_len=configuration["min_len"] if "min_len" in configuration else None,
            sort_by_size=configuration["sort_by_size"] if "sort_by_size" in configuration else None,
            num_chunks=configuration["num_chunks"],
            min_chunk_size=configuration["min_chunk_size"],
            is_training=True
        )
        if configuration["sort_by_size"] is True:
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=configuration["batch_size"],
                num_workers=num_workers,
                collate_fn=training_dataset.pad_collate_func,
                sampler=RandomChunkSampler(training_dataset, configuration["batch_size"]))
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=1,
                num_workers=num_workers,
                collate_fn=validation_dataset.pad_collate_func,
                sampler=RandomChunkSampler(validation_dataset, 1)
            )
        else:
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=configuration["batch_size"],
                shuffle=True,
                num_workers=num_workers,
                collate_fn=training_dataset.pad_collate_func)
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=validation_dataset.pad_collate_func
            )
    elif configuration["dataset_type"] == "Grayscale":
        training_dataset = GrayscaleDataset(
            csv_filepath=configuration["training_file"],
            width=configuration["width"],
            height=configuration["height"],
            convert_to_3d_image=configuration["convert_to_3d_image"],
        )
        validation_dataset = GrayscaleDataset(
            csv_filepath=configuration["validation_file"],
            width=configuration["width"],
            height=configuration["height"],
            convert_to_3d_image=configuration["convert_to_3d_image"],
        )
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=configuration["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
        )

    else:
        raise ValueError(
            f"Dataset type {configuration['dataset_type']} not found. Please use one of the following: Binary, RS, RsDel, DRS, SequentialDRS, RandomDRS, Grayscale"
        )
    return (
        training_dataset,
        validation_dataset,
        training_dataloader,
        validation_dataloader,
    )


def save_results(trainer: EarlyStoppingPyTorchTrainer, configuration: dict):
    results = {
        "training_losses": trainer.training_losses,
        "training_accuracies": trainer.training_accuracies,
        "validation_losses": trainer.validation_losses,
        "validation_accuracies": trainer.validation_accuracies,
    }
    with open(
        os.path.join(configuration["model_path"], "results.json"), "w"
    ) as output_file:
        json.dump(results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train end2end malware detector")
    parser.add_argument(
        "configuration_file",
        type=str,
        help="JSON-like file including the training and model configuration hyperparameters",
    )

    args = parser.parse_args()

    configuration = read_json_file(args.configuration_file)

    training_dataset, validation_dataset, training_dataloader, validation_dataloader = (
        create_datasets(configuration)
    )

    model = build_model(configuration)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=configuration["lr"] if configuration["lr"] else 1e-3)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([configuration["pos_weight"]]) if "pos_weight" in configuration else torch.Tensor([1.0])).to(device)
    trainer = EarlyStoppingPyTorchTrainer(
        optimizer, configuration["num_epochs"], criterion
    )
    model = trainer.train(
        model, training_dataloader, validation_dataloader, configuration["patience"]
    )
    if not os.path.exists(configuration["model_path"]):
        os.makedirs(configuration["model_path"])
    torch.save(
        model.state_dict(), os.path.join(configuration["model_path"], "model.pth")
    )
    save_results(trainer, configuration)
