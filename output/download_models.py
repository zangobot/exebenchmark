import sys
sys.path.append("../")
from maltorch.utils.utils import download_gdrive
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download models')
    parser.add_argument("input_filepath",
                        type=str,
                        help="CSV-like file containing the filepaths where you want to download the models and their associated Google Drive links")
    args = parser.parse_args()

    with open(args.input_filepath, "r") as input_file:
        lines = input_file.readlines()
        for line in lines:
            line = line.strip()
            filepath = line[0]
            gdrive_link = line[1]
            directory = "/".join(filepath.split("/")[:-1])
            if not os.path.exists(directory):
                os.makedirs(directory)
            download_gdrive(gdrive_id=gdrive_link, fname_save=filepath)

