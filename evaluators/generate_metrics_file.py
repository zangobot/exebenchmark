import argparse
import sys
sys.path.append("../")
from utils import write_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument("input_file",
                        type=str,
                        help="File with the scores and true label")
    parser.add_argument("output_file",
                        type=str,
                        help="Where to store the results")
    args = parser.parse_args()

    y_preds = []
    y_trues = []
    with open(args.input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split(",")
            y_score, y_true = tokens[0], tokens[1]
            y_pred = 1 if float(y_score) > 0.5 else 0
            y_preds.append(int(y_pred))
            y_trues.append(int(y_true))

    write_metrics(y_preds, y_trues, args.output_file)


