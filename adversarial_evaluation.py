import multiprocessing

from evaluators.adv_evaluator import AdversarialEvaluator
from utils import read_json_file
import os
import time
import torch
from maltorch.adv.evasion.base_optim_attack_creator import OptimizerBackends

if __name__ == "__main__":


    # config file where to define models and attacks 
    macro_config = read_json_file("configurations/ADVERSARIAL/adversarial.json")
    models = macro_config["models"]
    attacks = macro_config["attacks"]

    for attack, param in attacks.items():
        for model in models:
            
            print(f"Computing Adversarials: {model} with attack: {attack}")

            attack_name = "_".join(attack.split("_")[:-1])  # keep everything but the last split

            if param == "OptimizerBackends.NG":
                # For nevergrad attacks, we use the NG backend
                param = OptimizerBackends.NG

            # Create the configuration for the micro-evaluation 
            micro_config = {
                "architecture": model,
                "attack": attack_name,
                "param": param,
                "examples_folder": "adversarial_evaluation/adversarial_examples",
                "predictions_path": "adversarial_evaluation/adversarial_scores/",
                "transfer_path": None
            }
            eval = AdversarialEvaluator(micro_config, device="cpu")
            eval.bulk_attack()


