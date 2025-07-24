from utils import read_json_file
import lightgbm as lgb
import argparse
from utils import load_ember_csv
import os
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train end2end malware detector")
    parser.add_argument(
        "configuration_file",
        type=str,
        help="JSON-like file including the training and model configuration hyperparameters",
    )
    args = parser.parse_args()
    configuration = read_json_file(args.configuration_file)

    # Load training and validation data
    X_train, y_train = load_ember_csv(configuration["training_file"])
    X_validation, y_validation = load_ember_csv(configuration["validation_file"])

    # Create LightGBM datasets
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=[])
    valid_dataset = lgb.Dataset(
        X_validation,
        label=y_validation,
        reference=train_dataset,
        categorical_feature=[],
    )

    params = {
        "objective": configuration["objective"],
        "num_iterations": configuration["num_iterations"],
        "learning_rate": configuration["learning_rate"],
        "num_leaves": configuration["num_leaves"],
        "feature_fraction": configuration["feature_fraction"],
        "bagging_fraction": configuration["bagging_fraction"],
        "max_depth": configuration["max_depth"],
    }
    start_time = time.time()
    # Train the model
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=100,
        valid_sets=[valid_dataset],
        valid_names=["eval"],
        early_stopping_rounds=10,
    )
    end_time = time.time()
    # Ensure the directory exists
    model_dir = os.path.dirname(configuration["model_path"])  # Extract directory path
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)  # Create directories if they don't exist
    # Save the model
    model.save_model(configuration["model_path"])
    with open(os.path.join(model_dir, "training_time.txt"), "w") as f:
        f.write("Training time: {}".format(end_time-start_time))
