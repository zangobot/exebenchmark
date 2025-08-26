import multiprocessing

from evaluators.adv_evaluator import AdversarialEvaluator
from utils import read_json_file
import os
import time
import torch
from maltorch.adv.evasion.base_optim_attack_creator import OptimizerBackends

macro_config = read_json_file("configurations/ADVERSARIAL/adversarial.json")
models = macro_config["models"]
attacks = macro_config["attacks_inference"]

attacks = list(attacks.items())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model in models:
    for attack, param in attacks:

        if model == "EmberGBDT":
            device = "cpu"

        print(f"Computing Transfer: {model} with attack: {attack}")

        attack = "_".join(attack.split("_")[:-1])  # keep everything but the last split

        if param == "OptimizerBackends.NEVERGRAD":
            param = OptimizerBackends.NG

        # Create the configuration for the micro-evaluation 
        micro_config = {
            "architecture": model,
            "attack": attack,
            "param": param,
            "examples_folder": "adversarial_evaluation/adversarial_examples/",
            "predictions_path": "adversarial_evaluation/adversarial_scores/",
            "transfer_path": "adversarial_evaluation/transfer_scores/"
        }
        eval = AdversarialEvaluator(micro_config, device=device, batch_size=1)
        eval.transfer_eval()

