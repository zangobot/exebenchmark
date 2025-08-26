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
* [adversarial_evaluation.py](adversarial_evaluation.py): Python script used to evaluate the robustness of the detectors against evasion attacks.
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
## Adversarial Evaluation
You can download the subset of nonpacked malicious executables used for the adversarial evaluation from https://drive.google.com/file/d/1qiShG-WUp-0itBPTAo8vvfUWF5dnJoE4/view?usp=sharing

In https://drive.google.com/file/d/1FgAojUDswwFpLvNypwJ9iaDk-C8XGKrz/view?usp=sharing you will find the families associated to each sample.

