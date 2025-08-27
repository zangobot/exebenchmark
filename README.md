# EXE-Bench: Ranking the Tradeoffs of AI-based Windows Malware Detectors for Real-World Usability

This repository contains the code for the experiments of the manuscript "EXE-Bench: Ranking the Tradeoffs of AI-based Windows Malware Detectors for Real-World Usability", submitted to USENIX'26.

EXE-Bench is a comprehensive benchmark of AI-based Windows malware detectors. EXE-Bench assesses performance, temporal and adversarial robustness, and computational overhead, aggregating them into a single score for direct and fair model comparison. 

## ML-based detectors
The detectors can be divided into two groups depending on their input: (1) feature-based detectors and (2) end-to-end detectors:
* Feature-based detectors: EMBER LightGBM
* End-to-end detectors: MalConv, AvastConv, BBDnn, NGramConv
* Image-based detectors: ResNet18

We implemented various defenses based on randomized and (de)randomized smoothing. These defenses have been applied only to end-to-end detectors.
- Randomized Smoothing (RS). This approach trains a malware detector by randomly masking bytes with probability p. Then, at inference time, it generates N masked versions, independently classifies each version, and aggregates the results through majority voting.
  - Link: https://link.springer.com/chapter/10.1007/978-3-031-54129-2_40, https://arxiv.org/pdf/2302.01757
- Randomized Deletions (RsDel).
This approach trains a malware detector by randomly deleting bytes with probability p. Then, at inference time, it generates N masked versions, independently classifies each version, and aggregates the results through majority voting.
  - Link: https://arxiv.org/pdf/2302.01757
- (De)Randomized Smoothing. Fixed-size Chunk DRS (F-DRS) and K-Partition DRS (K-DRS) approaches split the executable into chunks, independently classify each chunk, and aggregate results through majoirty voting. The approaches differ in the way the chunks are generated. 
F-DRS specifies a fixed chunk size, e.g. 32768 bytes. K-DRS specifies the number of partitions to extract per file, e.g., 12. 
  - F-DRS link: https://dl.acm.org/doi/abs/10.1145/3605764.3623914
  - K-DRS link: https://arxiv.org/pdf/2303.13372
- (De)Randomized Smoothing with Random Chunks (Random-DRS). During training, this chunk-based smoothing scheme trains a base classifier to make classifications on a subset of contiguous bytes or chunk of bytes. At test time, a large number of chunks are then classified by a base classifier and the consensus among these classifications is then reported as the final prediction. The random strategy randomly selects the locations of the chunks. The size of the chunks is defined as a percentage of the file size, e.g., 10%.
  - Link: https://ieeexplore.ieee.org/abstract/document/10506708 
- (De)Randomized Smoothing with Sequential Chunks (Sequential-DRS). Similar to Random-DRS, but it selects contiguous adjacent chunks instead of random chunks.
  - Link: https://ieeexplore.ieee.org/abstract/document/10506708

## Structure of the repository
* [adversarial_evaluation](adversarial_evaluation)/: Contains the adversarial scores and the transferability scores of the detectors
* [configurations](configurations)/: Contains the configurations of the detectors
* [evaluators](evaluators)/: Contains the implementation of the adversarial evaluator.
* [models](models)/: Contains the pretrained models
* [output](output)/: Contains the training, validation, and test results for each model.
* [results](results)/: Contains the results of the performance, adversarial evaluation, concept drift evaluation and inference times of the detectors.
* [training_splits](training_splits)/: Contains the training, validation and test splits into which the EMBER dataset was divided.
* [train_ember_lightgbm_detector.py](train_ember_lightgbm_detector.py): Python script used to train the LightGBM detector.
* [train_end2end_detector.py](train_end2end_detector.py): Python script used to train the end2end detectors.
* [evaluate_ember_lightgbm_detector.py](evaluate_ember_lightgbm_detector.py): Python script used to evaluate the LightGBM detector.
* [evaluate_end2end_detector.py](evaluate_end2end_detector.py): Python script used to evaluate the end2end detectors.
* [adversarial_evaluation.py](adversarial_evaluation.py): Python script used to compute adversarial attacks.
* [transfer_evaluation.py](transfer_evaluation.py): Python script used to evaluate the transferability of attacks.
* [temporal_evaluation.py](temporal_evaluation.py): Python script used to evaluate the detectors against concept drift.

## Installation
Create a virtualenvironment and install the corresponding [requirements](requirements.txt).
```
maltorch
git+https://github.com/zangobot/ember.git
tqdm
ruff
matplotlib
SciencePlots
```




## Training/Evaluation
Various scripts are provided to train and evaluate the end-to-end and feature-based models as follows:
- [train_ember_lightgbm_detector.py](train_ember_lightgbm_detector.py)
- [train_end2end_detector.py](train_end2end_detector.py)
- [evaluate_ember_lightgbm_detector.py](evaluate_ember_lightgbm_detector.py)
- [evaluate_end2end_detector.py](evaluate_end2end_detector.py)

To train/evaluate the models you need to provide a configuration file with the hyperparameters, model path, etcetera. You can find examples in the [configurations](configurations) folder.

Example of training configuration file:
```
{
  "training_file": "training_splits/ember/ember_training_file.txt",
  "validation_file": "training_splits/ember/ember_validation_file.txt",
  "test_file": "training_splits/ember/ember_test_file.txt",
  "batch_size": 64,
  "num_epochs": 100,
  "patience": 5,
  "architecture": "MalConv",
  "embedding_size": 8,
  "max_len": 2000000,
  "min_len": 512,
  "threshold": 0.5,
  "padding_idx": 256,
  "model_path": "output/EMBER-MalConv-2000000",
  "dataset_type": "Binary"
}
```

Example of evaluation configuration file:
```
{
  "evaluation_file": "training_splits/ember/ember_test_file.txt",
  "batch_size": 1,
  "architecture": "MalConv",
  "embedding_size": 8,
  "max_len": 2000000,
  "min_len": 512,
  "threshold": 0.5,
  "padding_idx": 256,
  "model_path": "output/EMBER-MalConv-2000000/model.pth",
  "predictions_path": "output/EMBER-MalConv-2000000/EMBER_test_set_predictions.csv",
  "metrics_path": "output/EMBER-MalConv-2000000/EMBER_test_set_metrics.out"
}
```

Training/validation example:
```
python train_end2end_detector.py configurations/EMBER/training/MalConv/malconv_ember_configuration_file_pos_weight_0.875.json
python evaluate_end2end_detector.py configurations/EMBER/validation/MalConv/malconv_ember_validation_set_configuration_file_pos_weight_0.875.json
python evaluate_end2end_detector.py configurations/EMBER/test/MalConv/malconv_ember_test_set_configuration_file_pos_weight_0.875.json

```

## Temporal Evaluation
The temporal evaluation can be performed using the script [temporal_evaluation.py](temporal_evaluation.py). 
The script loads the configuration file [temporal_analysis.json](configurations/SPEAKEASY/temporal_analysis.json), containing the models, the temporal bins to test, the path of 
the [Speakeasy metadata](configurations/SPEAKEASY/all_metadata.csv) and the output path. The metadata files reports the hashes, the local path, the timestamp, the family and the labels of samples. In the uploaded file, the local path column is not specified and must be filled by the user.
The script uses the [evaluator interface](evaluators/evaluator.py), which initializes the models and performs inference.
The script uses GPU if available, besides when testing EmberGBDT.

## Adversarial and Transfer Evaluation
You can download the subset of nonpacked malicious executables used for the adversarial evaluation from https://drive.google.com/file/d/1qiShG-WUp-0itBPTAo8vvfUWF5dnJoE4/view?usp=sharing

In https://drive.google.com/file/d/1FgAojUDswwFpLvNypwJ9iaDk-C8XGKrz/view?usp=sharing you will find the families associated to each sample.

To perform attack computation against Vanilla models you can use the script [adversarial_evaluation.py](adversarial_evaluation.py). The script loads the configuration file [adversarial_evaluation.json](configurations/ADVERSARIAL/adversarial.json), containing the models to attack and the attack with parameters to use. 
The script then specifies which model to attack with the selected manipulation and parameters, where to save the adversarial examples and the corresponding adversarial scores to the [adversarial evaluator](evaluators/adv_evaluator.py), which initializes the models and performs the attacks.
Attacks can be run in parallel specifying the number of workers inside the script. It uses only one job if not specified. 
For GAMMA attacks, the path to the goodware and malware folders must be specified inside [config.py](config.py).

We computed adversarial examples using GAMMA (https://ieeexplore.ieee.org/abstract/document/9437194), FullDOS (https://arxiv.org/pdf/2008.07125) and Content-Shift (https://arxiv.org/pdf/2008.07125) attacks. GAMMA uses goodware harvested from Windows 11 system files or 
from the Speakeasy dataset. When injecting Windows sections the attack is initialized with 5, 10, 20, 30, and 50 sections, while with 5 and 10 
sections when injecting Speakeasy sections.

For the transfer evaluation, the same configuration file is used by [transfer_evaluation.py](transfer_evaluation.py) to test all models against the adversarial attacks previously computed. In this case, the script specifies also where to save the transfer scores to the [adversarial evaluator](evaluators/adv_evaluator.py).
The script uses GPU if available, besides when testing EmberGBDT.

The adversarial evaluator expects the following structure for the configuration file: 
```
{
  "architecture": model,
  "attack": attack,
  "param": param,
  "examples_folder": "adversarial_evaluation/adversarial_examples/",
  "predictions_path": "adversarial_evaluation/adversarial_scores/",
  "transfer_path": "adversarial_evaluation/transfer_scores/"
}
```
Both [adversarial_evaluation.py](adversarial_evaluation.py) and [transfer_evaluation.py](transfer_evaluation.py) use the [adversarial_evaluation.json](configurations/ADVERSARIAL/adversarial.json) to create each time a different configuration to pass to the adversarial evaluator.

The adversarial evaluator uses the interface [evaluator](evaluators/evaluator.py), to initialize the models. 
