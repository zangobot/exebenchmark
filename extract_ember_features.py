import argparse
from ember.features import PEFeatureExtractor
import csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train end2end malware detector")
    parser.add_argument(
        "hashes_filepath",
        type=str,
        help="CSV-like file containing the path to the files and their corresponding labels",
    )
    parser.add_argument(
        "output_filepath", type=str, help="CSV-like file containing the EMBER features"
    )
    args = parser.parse_args()

    feature_extractor = PEFeatureExtractor()
    with open(args.hashes_filepath, "r") as f:
        data = []
        for line in f.readlines():
            filepath, label = line.strip().split(",")
            data.append([filepath, label])

    with open(args.output_filepath, "w") as f:
        writer = csv.writer(f)
        i = 1
        for filepath, label in data:
            print("{},{},{}".format(i, filepath, label))
            with open(filepath, "rb") as bytez_file:
                bytez = bytez_file.read()
            features = feature_extractor.feature_vector(bytez)
            features = list(features) + [int(label)]
            writer.writerow(features)
            i += 1
