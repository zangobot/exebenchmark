import argparse
from utils import read_json_file, load_ember_csv, write_predictions, write_metrics
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import lightgbm as lgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train end2end malware detector')
    parser.add_argument("configuration_file",
                        type=str,
                        help="JSON-like file including the training and model configuration hyperparameters")
    args = parser.parse_args()

    configuration = read_json_file(args.configuration_file)
    X_eval, y_eval = load_ember_csv(configuration["evaluation_file"])

    # Load the trained LightGBM model
    model = lgb.Booster(model_file=configuration["model_path"])

    # Make predictions
    y_pred_prob = model.predict(X_eval)  # Predicted probabilities
    y_pred = (y_pred_prob > 0.5).astype(int)

    write_predictions(y_pred, y_eval, configuration["predictions_path"])
    write_metrics(y_pred, y_eval, configuration["metrics_path"])
