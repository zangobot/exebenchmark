import multiprocessing

from evaluators.adv_evaluator import AdversarialEvaluator
from utils import read_json_file
import os
import time
import torch
from maltorch.adv.evasion.base_optim_attack_creator import OptimizerBackends

macro_config = read_json_file("configurations/ADVERSARIAL/adversarial.json")
models = macro_config["models"][7:8]
attacks = macro_config["attacks_inference"]

attacks = list(attacks.items())

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

for model in models:
    for attack, param in attacks:

        # if (model == "EmberGBDT" or model=="ResNet18") and param == "OptimizerBackends.GRADIENT":
        #     # EmberGBDT does not support gradient attacks
        #     continue 

        if model == "EmberGBDT": 
            device = "cpu"
        else:
            device = "cuda:1"
        

        print(f"Computing Transfer: {model} with attack: {attack}")

        attack = "_".join(attack.split("_")[:-1])  # keep everything but the last split

        # if model == "MalConv" and attack == "content_shift":
        #     # Content shift attack is not supported by MalConv
        #     continue
        
        if param == "OptimizerBackends.NEVERGRAD":
            # For nevergrad attacks, we use the NG backend
            param = OptimizerBackends.NG
        elif param == "OptimizerBackends.GRADIENT":
            # For gradient attacks, we use the GRADIENT backend
            param = OptimizerBackends.GRADIENT

        # Create the configuration for the micro-evaluation 
        micro_config = {
            "architecture": model,
            "attack": attack,
            "param": param,
            "examples_folder": "adversarial_evaluation/adversarial_examples_1/",
            "predictions_path": "adversarial_evaluation/adversarial_scores_1/",
            "transfer_path": "adversarial_evaluation/transfer_scores_1/"
        }
        eval = AdversarialEvaluator(micro_config, device=device, batch_size=1)
        eval.transfer_eval()

