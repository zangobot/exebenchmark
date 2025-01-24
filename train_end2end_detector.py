import argparse
import torch
import json
from maltorch.datasets.binary_dataset import BinaryDataset
from maltorch.trainers.early_stopping_pytorch_trainer import EarlyStoppingPyTorchTrainer
from torch.utils.data import DataLoader
import multiprocessing
import os


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

def read_json_file(filepath: str)-> dict:
    with open(filepath, "r") as input_file:
        return json.load(input_file)

def build_model(configuration: dict) -> torch.nn.Module:
    architecture_name = configuration["architecture"]
    if architecture_name == "MalConv":
        from maltorch.zoo.malconv import MalConv
        return MalConv(
            embedding_size=configuration["embedding_size"],
            max_input_size=configuration["max_input_size"],
            threshold=configuration["threshold"],
            padding_value=configuration["padding_value"]
        )
    elif architecture_name == "AvastConv":
        from maltorch.zoo.avaststyleconv import AvastStyleConv
        return AvastStyleConv(
            embedding_size=configuration["embedding_size"],
            max_input_size=configuration["max_input_size"],
            threshold=configuration["threshold"],
            padding_value=configuration["padding_value"]
        )
    elif architecture_name == "NGramConv":
        from maltorch.zoo.ngramconv import NGramConv
        return NGramConv(
            embedding_size=configuration["embedding_size"],
            max_input_size=configuration["max_input_size"],
            threshold=configuration["threshold"],
            padding_value=configuration["padding_value"]
        )
    elif architecture_name == "ShallowConv":
        from maltorch.zoo.shallowconv import ShallowConv
        return ShallowConv(
            embedding_size=configuration["embedding_size"],
            max_input_size=configuration["max_input_size"],
            threshold=configuration["threshold"],
            padding_value=configuration["padding_value"]
        )
    elif architecture_name == "BBDnn":
        from maltorch.zoo.bbdnn import BBDnn
        return BBDnn() # Make sure to use padding_value = 0 for the BBDnn model
    else:
        raise ValueError(f"Model {architecture_name} not found")

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

    training_dataset = BinaryDataset(
        csv_filepath=configuration["training_file"],
        max_len=configuration["max_input_size"],
        padding_value=configuration["padding_value"]
    )
    validation_dataset = BinaryDataset(
        csv_filepath=configuration["validation_file"],
        max_len=configuration["max_input_size"],
        padding_value=configuration["padding_value"]
    )
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





