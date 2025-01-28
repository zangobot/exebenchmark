import json
import torch

def read_json_file(filepath: str)-> dict:
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