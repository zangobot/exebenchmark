import multiprocessing
from evaluators.adv_evaluator import AdversarialEvaluator
from utils import read_json_file
import os
import time
import argparse
from utils import check_cuda


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models against adv. attacks")
    parser.add_argument(
        "configuration_file",
        type=str,
        help="JSON-like file including the training and model configuration hyperparameters",
    )
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # config file where to define models and attacks 
    macro_config = read_json_file(args.configuration_file)
    models = macro_config["models"]
    attacks = macro_config["attacks"]

    # Set the multiprocessing start method to "spawn" for compatibility with linux machines
    multiprocessing.set_start_method("spawn")
    
    # Number of jobs to run in parallel, leaving 4 cores free for the system
    n_jobs = multiprocessing.cpu_count() - 4
    n_jobs = 1
    #print(multiprocessing.cpu_count())
    #n_jobs = 3
    print("Models: ", models)
    print("Attacks: ", attacks)
    for model in models:
        for attack in attacks:
            print(f"Computing Adversarials: {model} with attack: {attack}")
            
            # Create the configuration for the micro-evaluation 
            micro_config = {
                "architecture": model,
                "attack": attack,
                "examples_folder": macro_config.get("examples_folder", "adversarial_evaluation/adversarial_examples/"),
                "predictions_path": macro_config.get("predictions_path", "adversarial_evaluation/adversarial_scores/"),
            }
            # with small vanilla models swapping things in gpu is not worth it
            device = "cpu"
            eval = AdversarialEvaluator(micro_config, device=device)
            eval.create_attack()
            start = time.time()
            eval.bulk_attack(n_jobs=n_jobs)
            end = time.time()
            elapsed_hours = (end - start) / 3600
            # times are just for fun, becuase we use different machines, just for fun
            with open("adversarial_evaluation/times.txt", "a") as f:
                f.write(f"{model},{attack},{elapsed_hours:.4f}\n")


    # Once we have all advarsarial for all models, we can do trasfer evaluation
    
    # for model in models:
    #     for attack in attacks:
    #         print(f"Evaluating Adversarial Examples: {model} with attack: {attack}")
    #         micro_config = {
    #             "architecture": model,
    #             "attack": attack,
    #             "examples_folder": "adversarial_evaluation/adversarial_examples",
    #             "predictions_path": "adversarial_evaluation/adversarial_scores/",
    #             "transfer_path": "adversarial_evaluation/transfer_scores/"
    #         }
    #         eval = AdversarialEvaluator(micro_config)
    #         eval.transfer_eval()
