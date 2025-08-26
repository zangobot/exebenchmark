import os

from evaluators.evaluator import Evaluator
from utils import read_json_file


if __name__ == "__main__":
    macro_config = read_json_file("configurations/SPEAKEASY/temporal_analysis.json")

    for model in macro_config["models"]:
        for temporal_bin in macro_config["temporal_bins"]:
            micro_config = {
                "architecture": model,
                "predictions_folder": os.path.join(
                    macro_config["predictions_folder"], temporal_bin["temporal_bin"]
                ),
                "max_date": temporal_bin["max_date"],
                "min_date": temporal_bin["min_date"],
                "metadata_path": "configurations/SPEAKEASY/all_metadata.csv",
            }

            print(
                f"Evaluating {micro_config['architecture']} "
                f"Period: {temporal_bin['temporal_bin']}"
            )

            evaluator = Evaluator(config=micro_config)
            evaluator.evaluate(
                batch_size=1
            ) 
