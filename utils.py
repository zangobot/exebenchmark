import json
import torch
import numpy as np
import csv

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

def load_ember_csv(filepath: str) -> np.array:
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