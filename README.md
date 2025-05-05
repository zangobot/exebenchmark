# EXEBenchmark
This repository contains the code for the experiments in the paper "EXE-Benchmark".

# Installation
Clone maltorch's repository (https://github.com/zangobot/maltorch) and install the dependencies listed in the requirements.txt file.

## ML-based Models
Following you can find the machine learning-based models trained and its Google Drive link to download them.

- MalConv (https://cdn.aaai.org/ocs/ws/ws0432/16422-75958-1-PB.pdf)
  - Vanilla (Download to exebenchmark/output/EMBER-MalConv-2000000-pos-weight-0.875/): https://drive.google.com/file/d/1Uk7QHjjXMEy-RADX5kHD9vIYk6UT2nii/view?usp=sharing
    - architecture: MalConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512
    - kernel_size: 512,
    - stride: 512,
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: None
    - postprocessing: Sigmoid
    - dataset_type: Binary (for training only)
  - RS (Download to exebenchmark/output/EMBER-RS-MalConv-2000000-pos-weight-0.875/): https://drive.google.com/file/d/1xi-Dc758WxpuNNkfVJdpAGqEzd9Dx-rS/view?usp=sharing
    - architecture: MalConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512
    - kernel_size: 512,
    - stride: 512,
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: RS
    - pabl: 0.20
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RS (for training only)
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-MalConv-2000000-pos-weight-0.875/): https://drive.google.com/file/d/1Ste3BBC5eONw42-tih2zjhm4Rj0ck-P6/view?usp=sharing
    - architecture: MalConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512
    - kernel_size: 512,
    - stride: 512,
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: RsDel
    - pdel: 0.03
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RsDel (for training only)
  - Random-DRS. 
    - Random-DRS-0.05 (Download to exebenchmark/output/EMBER-Random-DRS-MalConv-pos-weight-0.05): https://drive.google.com/file/d/1W-BK9L1_wOnHqQ69XL-oc5feT_ewlRBv/view?usp=sharing
      - architecture: MalConv
      - embedding_size: 8
      - max_len: 2000000
      - min_len: 512
      - kernel_size: 512,
      - stride: 512,
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.05,
      - num_chunks: 100,
      - min_chunk_size: 512,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
    - Random-DRS-0.10: (Download to exebenchmark/output/EMBER-Random-DRS-MalConv-pos-weight-0.10): https://drive.google.com/file/d/1kKjuP-1a3Jy5BbbTZHXbHpmSw9jITCtY/view?usp=sharing
      - architecture: MalConv
      - embedding_size: 8
      - max_len: 2000000
      - min_len: 512
      - kernel_size: 512,
      - stride: 512,
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.10,
      - num_chunks: 100,
      - min_chunk_size: 512,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
    - Random-DRS-0.20: (Download to exebenchmark/output/EMBER-Random-DRS-MalConv-pos-weight-0.20): https://drive.google.com/file/d/1fqYsOB7n-dfbpd_LepooqYhon87_AgbY/view?usp=sharing
      - architecture: MalConv
      - embedding_size: 8
      - max_len: 2000000
      - min_len: 512
      - kernel_size: 512,
      - stride: 512,
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.20,
      - num_chunks: 100,
      - min_chunk_size: 512,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
  - DRS (Download to exebenchmark/output/EMBER-DRS-MalConv-200000/): 
  - Sequential-DRS (Download to exebenchmark/output/EMBER-Sequential-DRS-MalConv-2000000/): 
- AvastConv (https://openreview.net/pdf?id=HkHrmM1PM)
  - Vanilla (Download to exebenchmark/output/EMBER-AvastConv-512000-pos-weight-0.875/): https://drive.google.com/file/d/1THfGkbWemzhdzv6HESbwd9qY_bQ0_sub/view?usp=sharing
    - architecture: AvastConv
    - embedding_size: 8
    - max_len: 512000
    - min_len: 10244
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: None
    - postprocessing: Sigmoid
    - dataset_type: Binary (for training only)
  - RS (Download to exebenchmark/output/EMBER-RS-AvastConv-512000-pos-weight-0.875/): https://drive.google.com/file/d/1KJ4dKIjJeIRX5Th4mUxh2sV5xtctwpPk/view?usp=sharing
    - architecture: AvastConv
    - embedding_size: 8
    - max_len: 512000
    - min_len: 10244
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: RS
    - pabl: 0.20
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RS (for training only)
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-AvastConv-512000-pos-weight-0.875/): https://drive.google.com/file/d/1RUvXAGzbPRfWrI1Bg-N-ZVAYoo3tTDv1/view?usp=sharing
    - architecture: AvastConv
    - embedding_size: 8
    - max_len: 512000
    - min_len: 10244
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: RsDel
    - pabl: 0.03
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RsDel (for training only)
  - Random-DRS. 
    - Random-DRS-0.05 (Download to exebenchmark/output/EMBER-Random-DRS-AvastConv-pos-weight-0.05): https://drive.google.com/file/d/1YEyF3bF4I2QPpVOr6dJDSZO8lgL3nSG2/view?usp=sharing
      - architecture: AvastConv
      - embedding_size: 8
      - max_len: 2000000
      - min_len: 10244
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.05,
      - num_chunks: 100,
      - min_chunk_size: 10244,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
    - Random-DRS-0.10: (Download to exebenchmark/output/EMBER-Random-DRS-AvastConv-pos-weight-0.10): https://drive.google.com/file/d/1IjyDBBu_7SrSy7ZEkcOCduQnsz_p5rvL/view?usp=sharing
      - architecture: AvastConv
      - embedding_size: 8
      - max_len: 2000000
      - min_len: 10244
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.10,
      - num_chunks: 100,
      - min_chunk_size: 10244,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
    - Random-DRS-0.20: (Download to exebenchmark/output/EMBER-Random-DRS-AvastConv-pos-weight-0.20): https://drive.google.com/file/d/15-Xs4zIhMgIyhR_WSzHwAmjVnS4oF6aL/view?usp=sharing
      - architecture: AvastConv
      - embedding_size: 8
      - max_len: 2000000
      - min_len: 10244
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.20,
      - num_chunks: 100,
      - min_chunk_size: 10244,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
  - DRS (Download to exebenchmark/output/EMBER-DRS-AvastConv-200000/): Needs To be Retrained
  - Sequential-DRS (Download to exebenchmark/output/EMBER-Sequential-DRS-AvastConv-2000000/): Needs To be Retrained
- BBDnn (https://ieeexplore.ieee.org/document/8844623)
  - Vanilla (Download to exebenchmark/output/EMBER-BBDnn-102400-pos-weight-0.875/): https://drive.google.com/file/d/1c_9lVHT9zYpBCwQfnUW6ZbCF6SaVabRZ/view?usp=sharing
    - architecture: BBDnn
    - embedding_size: 10
    - max_len: 102400
    - min_len: 4096
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: None
    - postprocessing: Sigmoid
    - dataset_type: Binary (for training only)
  - RS (Download to exebenchmark/output/EMBER-RS-BBDnn-102400-pos-weight-0.875/): https://drive.google.com/file/d/1lpsnFz7hFSrLXrguxRuu5Mrmarp9uNod/view?usp=sharing
    - architecture: BBDnn
    - embedding_size: 8
    - max_len: 102400
    - min_len: 4096
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: RS
    - pabl: 0.20
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RS (for training only)
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-BBDnn-102400-pos-weight-0.875/): https://drive.google.com/file/d/1Kq7HngMv4cVD0HjpaqHctqEpQlicOjW_/view?usp=sharing
    - architecture: BBDnn
    - embedding_size: 8
    - max_len: 102400
    - min_len: 4096
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: RsDel
    - pabl: 0.03
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RsDel (for training only)
  - Random-DRS. 
    - Random-DRS-0.05 (Download to exebenchmark/output/EMBER-Random-DRS-BBDnn-pos-weight-0.05): https://drive.google.com/file/d/1-RQmJrpW_-zmez5oM-9DDhbVSDHoZm_F/view?usp=sharing
      - architecture: BBDnn
      - embedding_size: 10
      - max_len: 512000
      - min_len: 4096
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.05,
      - num_chunks: 100,
      - min_chunk_size: 4096,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
    - Random-DRS-0.10: (Download to exebenchmark/output/EMBER-Random-DRS-BBDnn-pos-weight-0.10): https://drive.google.com/file/d/1rbpNqwVNwQpo1gS7olhXN0LYBOoH3NoC/view?usp=sharing
      - architecture: BBDnn
      - embedding_size: 10
      - max_len: 512000
      - min_len: 4096
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.10,
      - num_chunks: 100,
      - min_chunk_size: 4096,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
    - Random-DRS-0.20: (Download to exebenchmark/output/EMBER-Random-DRS-BBDnn-pos-weight-0.20): https://drive.google.com/file/d/1zC8s1r4MvUQtrbC8L3T95AoJXXyAvIZl/view?usp=sharing
      - architecture: BBDnn
      - embedding_size: 10
      - max_len: 512000
      - min_len: 4096
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.20,
      - num_chunks: 100,
      - min_chunk_size: 4096,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
  - DRS (Download to exebenchmark/output/EMBER-DRS-BBDnn-200000/): 
  - Sequential-DRS (Download to exebenchmark/output/EMBER-Sequential-DRS-BBDnn-2000000/): 
- NGramConv (https://www.sciencedirect.com/science/article/pii/S0167404820304326, https://ebooks.iospress.nl/doi/10.3233/978-1-61499-806-8-221)
  - Vanilla (Download to exebenchmark/output/EMBER-NGramConv-512000-pos-weight-0.875/): https://drive.google.com/file/d/1uTflFZSM1xE_Q_dDZ-n77Ud4CbtYFtlB/view?usp=sharing
    - architecture: NGramConv
    - embedding_size: 8
    - max_len: 512000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: None
    - postprocessing: Sigmoid
    - dataset_type: Binary (for training only)
  - RS (Download to exebenchmark/output/EMBER-RS-NGramConv-512000-pos-weight-0.875/): https://drive.google.com/file/d/1Zth04spqwN5ouxuTq6MzGdngiYrYXzLc/view?usp=sharing
    - architecture: BBDnn
    - embedding_size: 8
    - max_len: 512000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: RS
    - pabl: 0.20
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RS (for training only)
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-NGramConv-512000-pos-weight-0.875/): https://drive.google.com/file/d/1JQi9rgIvv3aRND2BDyD2mGXlmQnkSyDx/view?usp=sharing
    - architecture: BBDnn
    - embedding_size: 8
    - max_len: 512000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: RsDel
    - pabl: 0.03
    - num_versions: 100
    - postprocessing: MajorityVoting
    - dataset_type: RsDel (for training only)
  - Random-DRS. 
    - Random-DRS-0.05 (Download to exebenchmark/output/EMBER-Random-DRS-NGramConv-pos-weight-0.05): https://drive.google.com/file/d/1GNroJpdGF75gGVm9GnNmoaAI7XGCzuj8/view?usp=sharing
      - architecture: NGramConv
      - embedding_size: 8
      - max_len: 2000000
      - min_len: 512
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.05,
      - num_chunks: 100,
      - min_chunk_size: 512,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
    - Random-DRS-0.10: (Download to exebenchmark/output/EMBER-Random-DRS-NGramConv-pos-weight-0.10): https://drive.google.com/file/d/1jzc4hSvJmeY-83Vxqt5l7v_lTU39aBcC/view?usp=sharing
      - architecture: NGramConv
      - embedding_size: 8
      - max_len: 2000000
      - min_len: 512
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.10,
      - num_chunks: 100,
      - min_chunk_size: 512,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
    - Random-DRS-0.20: (Download to exebenchmark/output/EMBER-Random-DRS-NGramConv-pos-weight-0.20): https://drive.google.com/file/d/1yHDBHEuPIwKFmAjiunFGD8TYh4aWf83_/view?usp=sharing
      - architecture: NGramConv
      - embedding_size: 8
      - max_len: 2000000
      - min_len: 512
      - threshold: 0.5
      - padding_idx: 256
      - pos_weight: 0.875
      - preprocessing: RandomDRS
      - file_percentage: 0.20,
      - num_chunks: 100,
      - min_chunk_size: 512,
      - sort_by_size: true,
      - postprocessing: MajorityVoting
      - dataset_type: RandomDRS (for training only)
  - DRS (Download to exebenchmark/output/EMBER-DRS-NGramConv-200000/): 
  - Sequential-DRS (Download to exebenchmark/output/EMBER-Sequential-DRS-NGramConv-2000000/): 
- ResNet18 (Download to exebenchmark/output/EMBER-ResNet18-pos-weight-0.875/): https://drive.google.com/file/d/1N1uK8bsfJvB88ryZcbRfxkzerblXqMqg/view?usp=sharing
  - preprocessing: Grayscale
  - postprocessing: Sigmoid
  - dataset_type: Grayscale (for training only)
  - width = height = 256
  - convert_to_3d_image: true
  - threshold: 0.5
  - pos_weight: 0.875
- EMBER LightGBM: (Download to exebenchmark/output/EMBER-LightGBM/): https://drive.google.com/file/d/1RWvr3yD8M90EXcTozK2TwW2JEExQ9qDW/view?usp=sharing
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
  - pos_weight: 0.875

Papers describing the defenses:
- Randomized Smoothing (RS). This approach trains a malware detector by randomly masking bytes with probability p. Then, at inference time, it generates N masked versions, independently classifies each version, and aggregates the results through majority voting.
  - Link: https://link.springer.com/chapter/10.1007/978-3-031-54129-2_40
- Randomized Deletions (RsDel).
This approach trains a malware detector by randomly deleting bytes with probability p. Then, at inference time, it generates N masked versions, independently classifies each version, and aggregates the results through majority voting.
  - Link: https://arxiv.org/pdf/2302.01757
- (De)Randomized Smoothing (AISEC-DRS and ICLR-DRS). These approaches split the executable into chunks, independently classify each chunk, and aggregate results through majoirty voting. The approaches differ in the way the chunks are generated. 
AISEC-DRS specifies a fixed chunk size, e.g. 2048 bytes. ICLR-DRS specifies the number of chunks to extract per file, e.g., 4. 
  - AISEC-DRS link: https://dl.acm.org/doi/abs/10.1145/3605764.3623914
  - ICLR-DRS link: https://arxiv.org/pdf/2303.13372
- (De)Randomized Smoothing with Random Chunks (Random-DRS). During training, this chunk-based smoothing scheme trains a base classifier to make classifications on a subset of contiguous bytes or chunk of bytes. At test time, a large number of chunks are then classified by a base classifier and the consensus among these classifications is then reported as the final prediction. The random strategy randomly selects the locations of the chunks. The size of the chunks are defined as a percentage of the file size, e.g., 20%.
  - Link: https://ieeexplore.ieee.org/abstract/document/10506708 
- (De)Randomized Smoothing with Sequential Chunks (Sequential-DRS). Similar to Random-DRS, but it selects contiguous adjacent chunks instead of random chunks.
  - Link: https://ieeexplore.ieee.org/abstract/document/10506708

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


## Adversarial Evaluation
You can download a subset of nonpacked malicious executables from https://drive.google.com/file/d/1qiShG-WUp-0itBPTAo8vvfUWF5dnJoE4/view?usp=sharing

In https://drive.google.com/file/d/1FgAojUDswwFpLvNypwJ9iaDk-C8XGKrz/view?usp=sharing you will find the families associated to each sample.