# EXEBenchmark
This repository contains the code for the experiments in the paper "EXE-Benchmark".

# Installation
Clone maltorch's repository (https://github.com/zangobot/maltorch) and install the dependencies listed in the requirements.txt file.

## ML-based Models
Following you can find the machine learning-based models trained and its Google Drive link to download them.

- MalConv
  - Vanilla (Download to exebenchmark/output/EMBER-MalConv-2000000-pos-weight-0.875/): https://drive.google.com/file/d/1Uk7QHjjXMEy-RADX5kHD9vIYk6UT2nii/view?usp=sharing
    - architecture: MalConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: None
    - postprocessing: None
    - dataset_type: Binary (for training only)
  - RS (Download to exebenchmark/output/EMBER-RS-MalConv-2000000/): https://drive.google.com/file/d/1xi-Dc758WxpuNNkfVJdpAGqEzd9Dx-rS/view?usp=drive_link
    - architecture: MalConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - preprocessing: RS
    - pabl: 0.20
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RS (for training only)
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-MalConv-2000000/): https://drive.google.com/file/d/1Ste3BBC5eONw42-tih2zjhm4Rj0ck-P6/view?usp=drive_link
    - architecture: MalConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - preprocessing: RsDel
    - pdel: 0.03
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RsDel (for training only)
  - DRS (Download to exebenchmark/output/EMBER-DRS-MalConv-200000/): 
  - Sequential-DRS (Download to exebenchmark/output/EMBER-Sequential-DRS-MalConv-2000000/): 
  - Random-DRS (Download to exebenchmark/output/EMBER-Random-DRS-MalConv-2000000/): 
- AvastConv
  - Vanilla (Download to exebenchmark/output/EMBER-AvastConv-512000-pos-weight-0.875/): https://drive.google.com/file/d/14wSZQ-Drns9G8CEqvfbAZjmMVt_ToUbp/view?usp=sharing
    - architecture: AvastConv
    - embedding_size: 8
    - max_len: 512000
    - min_len: 10244
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: None
    - postprocessing: None
    - dataset_type: Binary (for training only)
  - RS (Download to exebenchmark/output/EMBER-RS-AvastConv-512000/): https://drive.google.com/file/d/1KJ4dKIjJeIRX5Th4mUxh2sV5xtctwpPk/view?usp=drive_link
    - architecture: AvastConv
    - embedding_size: 8
    - max_len: 512000
    - min_len: 10244
    - threshold: 0.5
    - padding_idx: 256
    - preprocessing: RS
    - pabl: 0.20
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RS (for training only)
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-AvastConv-512000/): https://drive.google.com/file/d/1RUvXAGzbPRfWrI1Bg-N-ZVAYoo3tTDv1/view?usp=drive_link
    - architecture: AvastConv
    - embedding_size: 8
    - max_len: 512000
    - min_len: 10244
    - threshold: 0.5
    - padding_idx: 256
    - preprocessing: RsDel
    - pabl: 0.03
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RsDel (for training only)
  - DRS (Download to exebenchmark/output/EMBER-DRS-AvastConv-200000/): 
  - Sequential-DRS (Download to exebenchmark/output/EMBER-Sequential-DRS-AvastConv-2000000/): 
  - Random-DRS (Download to exebenchmark/output/EMBER-Random-DRS-AvastConv-2000000/): 
- BBDnn
  - Vanilla (Download to exebenchmark/output/EMBER-BBDnn-102400-pos-weight-0.875/): https://drive.google.com/file/d/1c_9lVHT9zYpBCwQfnUW6ZbCF6SaVabRZ/view?usp=sharing
    - architecture: BBDnn
    - embedding_size: 8
    - max_len: 102400
    - min_len: 4096
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: None
    - postprocessing: None
    - dataset_type: Binary (for training only)
  - RS (Download to exebenchmark/output/EMBER-RS-BBDnn-102400/): https://drive.google.com/file/d/1lv53kNIIACWRPD8ncEY8To0S2JNN_AY8/view?usp=drive_link
    - architecture: BBDnn
    - embedding_size: 8
    - max_len: 102400
    - min_len: 4096
    - threshold: 0.5
    - padding_idx: 256
    - preprocessing: RS
    - pabl: 0.20
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RS (for training only)
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-BBDnn-102400/): https://drive.google.com/file/d/1Fl5DN6QxKWmO9JxNNFIiQI5FzhP6qC4y/view?usp=drive_link
    - architecture: BBDnn
    - embedding_size: 8
    - max_len: 102400
    - min_len: 4096
    - threshold: 0.5
    - padding_idx: 256
    - preprocessing: RsDel
    - pabl: 0.03
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RsDel (for training only)
  - DRS (Download to exebenchmark/output/EMBER-DRS-BBDnn-200000/): 
  - Sequential-DRS (Download to exebenchmark/output/EMBER-Sequential-DRS-BBDnn-2000000/): 
  - Random-DRS (Download to exebenchmark/output/EMBER-Random-DRS-BBDnn-2000000/): 
- NGramConv
  - Vanilla (Download to exebenchmark/output/EMBER-NGramConv-512000-pos-weight-0.875/): https://drive.google.com/file/d/1uTflFZSM1xE_Q_dDZ-n77Ud4CbtYFtlB/view?usp=sharing
    - architecture: NGramConv
    - embedding_size: 8
    - max_len: 512000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - preprocessing: None
    - postprocessing: None
    - dataset_type: Binary (for training only)
  - RS (Download to exebenchmark/output/EMBER-RS-NGramConv-512000/): https://drive.google.com/file/d/1LPDMv--V9ij9xndGN6zIKi1zf77fDzmH/view?usp=drive_link
    - architecture: BBDnn
    - embedding_size: 8
    - max_len: 512000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - preprocessing: RS
    - pabl: 0.20
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RS (for training only)
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-NGramConv-512000/): https://drive.google.com/file/d/1JQi9rgIvv3aRND2BDyD2mGXlmQnkSyDx/view?usp=drive_link
    - architecture: BBDnn
    - embedding_size: 8
    - max_len: 512000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - preprocessing: RsDel
    - pabl: 0.03
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RsDel (for training only)
  - DRS (Download to exebenchmark/output/EMBER-DRS-NGramConv-200000/): 
  - Sequential-DRS (Download to exebenchmark/output/EMBER-Sequential-DRS-NGramConv-2000000/): 
  - Random-DRS (Download to exebenchmark/output/EMBER-Random-DRS-NGramConv-2000000/):
- ResNet18 (Download to exebenchmark/output/EMBER-ResNet18/): https://drive.google.com/file/d/1N1uK8bsfJvB88ryZcbRfxkzerblXqMqg/view?usp=drive_link
  - preprocessing: Grayscale
  - postprocessing: None
  - dataset_type: Grayscale (for training only)
  - width = height = 256
  - convert_to_3d_image: true
  - threshold: 0.5
- EMBER LightGBM: (Download to exebenchmark/output/EMBER-LightGBM/): https://drive.google.com/file/d/1RWvr3yD8M90EXcTozK2TwW2JEExQ9qDW/view?usp=drive_link
  - objective: binary
  - num_iterations: 5000
  - learning_rate: 0.05
  - max_bin: 255
  - num_leaves: 31
  - pos_weight: 0.875
  - feature_fraction: 0.7
  - bagging_fraction: 0.7
  - max_depth: -1
  - boosting: gbdt
## EMBER Dataset
The EMBER dataset has been split into three sets: training, validation, and test. The original EMBER dataset contains 400,000 benign and 400,000 malicious samples. 
Unfortunately, I have only been able to retrieve 349,994/400,000 benign samples and 399,992/400,000 malicious samples.
The training set contains 599,998 samples, the validation set contains 74,997 samples, and the test set contains 74,999 samples. 
The training splits are available at:
- [ember_training_file_server.txt](training_splits/ember/ember_training_file_server.txt)
- [ember_validation_file_server.txt](training_splits/ember/ember_validation_file_server.txt)
- [ember_test_file_server.txt](training_splits/ember/ember_test_file_server.txt)

## Training and Evaluation Scripts
Various scripts are provided to train and evaluate the end-to-end and feature-based models as follows:
- [train_ember_lightgbm_detector.py](train_ember_lightgbm_detector.py)
- [train_end2end_detector.py](train_end2end_detector.py)
- [evaluate_ember_lightgbm_detector.py](evaluate_ember_lightgbm_detector.py)
- [evaluate_end2end_detector.py](evaluate_end2end_detector.py)

To train/evaluate the models you need to provide a configuration file with the hyperparameters, model path, etcetera. You can find examples in the [configurations](configurations) folder.
Example of training configuration file:
```
{
  "training_file": "training_splits/ember/ember_training_file_server.txt",
  "validation_file": "training_splits/ember/ember_validation_file_server.txt",
  "test_file": "training_splits/ember/ember_test_file_server.txt",
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
  "evaluation_file": "training_splits/ember/ember_test_file_server.txt",
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


You will find the scripts used for training/evaluation the models in the [scripts](scripts) folder.


