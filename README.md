# EXEBenchmark
This repository contains the code for the experiments in the paper "EXE-Benchmark".

# Installation
Clone maltorch's repository (https://github.com/zangobot/maltorch) and install the dependencies listed in the requirements.txt file.

## ML-based Models
Following you can find the machine learning-based models trained and its Google Drive link to download them.

- MalConv (https://cdn.aaai.org/ocs/ws/ws0432/16422-75958-1-PB.pdf)
  - Vanilla-MalConv (Download to exebenchmark/output/EMBER-MalConv-2000000-pos-weight-0.875/): 
    - Google Drive ID: 1Uk7QHjjXMEy-RADX5kHD9vIYk6UT2nii
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
  - RS-MalConv (Download to exebenchmark/output/EMBER-RS-MalConv-2000000-pos-weight-0.875/): 
    - Google Drive ID: 1xi-Dc758WxpuNNkfVJdpAGqEzd9Dx-rS
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
  - RsDel-MalConv (Download to exebenchmark/output/EMBER-RsDel-MalConv-2000000-pos-weight-0.875/): 
    - Google Drive ID: 1Ste3BBC5eONw42-tih2zjhm4Rj0ck-P6
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
  - Random-DRS-MalConv-0.10: (Download to exebenchmark/output/EMBER-Random-DRS-MalConv-pos-weight-0.10): 
    - Google Drive ID: 1npB96W3HE4aM8LtEHqOGXI9QRrpTJMoK
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
  - Sequential-DRS-MalConv-0.10: (Download to exebenchmark/output/EMBER-Sequential-DRS-MalConv-pos-weight-0.10): 
    - Google Drive ID: 1X5L5Wt2vZrM_MVkBmFcjXEFVtYNP5i1d
    - architecture: MalConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512
    - kernel_size: 512,
    - stride: 512,
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: SequentialDRS
    - file_percentage: 0.10,
    - num_chunks: 100,
    - min_chunk_size: 512,
    - sort_by_size: true,
    - postprocessing: MajorityVoting
    - dataset_type: SequentialDRS (for training only)
  - F-DRS-MalConv-32768
    - Google Drive ID: 1JDJNGjMjuJQ9wMqT_eu4Z8DMmUlCYqmh
    - architecture: MalConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 32768
    - kernel_size: 512,
    - stride: 512,
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: F-DRS
    - chunk_size: 32768,
    - sort_by_size: true,
    - postprocessing: MajorityVoting
    - dataset_type: F-DRS (for training only)
  - K-DRS-MalConv-12
    - Google Drive ID: 1WenoLadUblWfEIan-2n9l-lBk79cK7IA
    - architecture: MalConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512
    - kernel_size: 512,
    - stride: 512,
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: K-DRS
    - num_chunks: 12
    - postprocessing: MajorityVoting
- AvastConv (https://openreview.net/pdf?id=HkHrmM1PM)
  - Vanilla (Download to exebenchmark/output/EMBER-AvastConv-512000-pos-weight-0.875/): 
    - Google Drive ID: 1KOB-o-2avfPtsQaGuguRW-b1HDbX8r8v
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
  - RS (Download to exebenchmark/output/EMBER-RS-AvastConv-512000-pos-weight-0.875/): 
    - Google Drive ID: 1KJ4dKIjJeIRX5Th4mUxh2sV5xtctwpPk
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
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-AvastConv-512000-pos-weight-0.875/): 
    - Google Drive ID: 1RUvXAGzbPRfWrI1Bg-N-ZVAYoo3tTDv1
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
  - Random-DRS-0.10: (Download to exebenchmark/output/EMBER-Random-DRS-AvastConv-pos-weight-0.10): 
    - Google Drive ID: 1Q-y7SBhzH0BhUbAi9AjFxiMgWiKs7Zpo
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
  - Sequential-DRS-0.10: (Download to exebenchmark/output/EMBER-Sequential-DRS-AvastConv-pos-weight-0.10): 
    - Google Drive ID: 1Z35C4nZdJ4PKVj46YsOGU8rcrw89LCzc
    - architecture: AvastConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 10244
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: SequentialDRS
    - file_percentage: 0.10,
    - num_chunks: 100,
    - min_chunk_size: 10244,
    - sort_by_size: true,
    - postprocessing: MajorityVoting
    - dataset_type: SequentialDRS (for training only)
  - F-DRS-AvastConv-32768
    - Google Drive ID: 1y3ZDhwIuMyszTRgp-0BLaB7ty3LVVsJT
    - architecture: AvastConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 32768
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: F-DRS
    - chunk_size: 32768,
    - sort_by_size: true,
    - postprocessing: MajorityVoting
    - dataset_type: F-DRS (for training only)
  - K-DRS-AvastConv-12
    - Google Drive ID: 1_lsE6RYo0ZNq4lxYXMtQHEPvAbioI116
    - architecture: AvastConv
    - embedding_size: 8
    - max_len: 512000
    - min_len: 10244
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: K-DRS
    - num_chunks: 12
    - postprocessing: MajorityVoting
- BBDnn (https://ieeexplore.ieee.org/document/8844623)
  - Vanilla (Download to exebenchmark/output/EMBER-BBDnn-102400-pos-weight-0.875/): 
    - Google Drive ID: 1c_9lVHT9zYpBCwQfnUW6ZbCF6SaVabRZ
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
  - RS (Download to exebenchmark/output/EMBER-RS-BBDnn-102400-pos-weight-0.875/): 
    - Google Drive ID: 1lpsnFz7hFSrLXrguxRuu5Mrmarp9uNod
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
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-BBDnn-102400-pos-weight-0.875/): 
    - Google Drive ID: 1Kq7HngMv4cVD0HjpaqHctqEpQlicOjW_
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
  - Random-DRS-0.10: (Download to exebenchmark/output/EMBER-Random-DRS-BBDnn-pos-weight-0.10): 
    - Google Drive ID: 1qnN_EZS_j5pxbfP_p-O27peUesLCj7OX
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
  - Sequential-DRS-0.10: (Download to exebenchmark/output/EMBER-Sequential-DRS-BBDnn-pos-weight-0.10): 
    - Google Drive ID: 1NMPXuiYhCJ_DNIQEHlD-NAl9-FcctGlw
    - architecture: BBDnn
    - embedding_size: 10
    - max_len: 512000
    - min_len: 4096
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: SequentialDRS
    - file_percentage: 0.10,
    - num_chunks: 100,
    - min_chunk_size: 4096,
    - sort_by_size: true,
    - postprocessing: MajorityVoting
    - dataset_type: SequentialDRS (for training only)
  - F-DRS-BBDnn-32768
    - Google Drive ID: 1ge_AB1_4CeAi6g0W38XtUlJAlHLDKrIv
    - architecture: BBDnn
    - embedding_size: 10
    - max_len: 2000000
    - min_len: 32768
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: F-DRS
    - chunk_size: 32768,
    - sort_by_size: true,
    - postprocessing: MajorityVoting
    - dataset_type: F-DRS (for training only)
  - K-DRS-BBDnn-12
    - Google Drive ID: 1v4LAaP_k2DFL8pjWwsexdlNmQLRDDJes
    - architecture: BBDnn
    - embedding_size: 10
    - max_len: 512000
    - min_len: 4096
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: K-DRS
    - num_chunks: 12
    - postprocessing: MajorityVoting
- NGramConv (https://www.sciencedirect.com/science/article/pii/S0167404820304326, https://ebooks.iospress.nl/doi/10.3233/978-1-61499-806-8-221)
  - Vanilla (Download to exebenchmark/output/EMBER-NGramConv-512000-pos-weight-0.875/): 
    - Google Drive ID: 1uTflFZSM1xE_Q_dDZ-n77Ud4CbtYFtlB
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
  - RS (Download to exebenchmark/output/EMBER-RS-NGramConv-512000-pos-weight-0.875/): 
    - Google Drive ID: 1Zth04spqwN5ouxuTq6MzGdngiYrYXzLc
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
  - RsDel (Download to exebenchmark/output/EMBER-RsDel-NGramConv-512000-pos-weight-0.875/): 
    - Google Drive ID: 1Ste3BBC5eONw42-tih2zjhm4Rj0ck-P6
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
  - Random-DRS-0.10: (Download to exebenchmark/output/EMBER-Random-DRS-NGramConv-pos-weight-0.10):
    - Google Drive ID: 1npB96W3HE4aM8LtEHqOGXI9QRrpTJMoK
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
  - Sequential-DRS-0.10: (Download to exebenchmark/output/EMBER-Sequential-DRS-NGramConv-pos-weight-0.10):
    - Google Drive ID: 1tNPoLYdYcFcWldc2QIOqBJzCjMND6EAy
    - architecture: NGramConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: SequentialDRS
    - file_percentage: 0.10,
    - num_chunks: 100,
    - min_chunk_size: 512,
    - sort_by_size: true,
    - postprocessing: MajorityVoting
    - dataset_type: SequentialDRS (for training only)
  - F-DRS-NGramConv-32768
    - Google Drive ID: 1DHa-Od-MPJUYSafSH6gybrX83T8xcQX_
    - architecture: NGramConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 32768
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: F-DRS
    - chunk_size: 32768,
    - sort_by_size: true,
    - postprocessing: MajorityVoting
    - dataset_type: F-DRS (for training only)
  - K-DRS-NGramConv-12
    - Google Drive ID: 1eVIu-2kD6UX_UmCOuVZa9pLeLOYWfK16
    - architecture: NGramConv
    - embedding_size: 8
    - max_len: 2000000
    - min_len: 512 ,
    - threshold: 0.5
    - padding_idx: 256
    - pos_weight: 0.875
    - preprocessing: K-DRS
    - num_chunks: 12
    - postprocessing: MajorityVoting
- ResNet18 (Download to exebenchmark/output/EMBER-ResNet18-pos-weight-0.875/):
  - Google Drive ID: 1N1uK8bsfJvB88ryZcbRfxkzerblXqMqg
  - preprocessing: Grayscale
  - postprocessing: Sigmoid
  - dataset_type: Grayscale (for training only)
  - width = height = 256
  - convert_to_3d_image: true
  - threshold: 0.5
  - pos_weight: 0.875
- EMBER LightGBM: (Download to exebenchmark/output/EMBER-LightGBM/): 
  - Google Drive ID: 1RWvr3yD8M90EXcTozK2TwW2JEExQ9qDW
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
- (De)Randomized Smoothing. Fixed-size Chunk DRS (F-DRS) and K-Partition DRS (K-DRS) approaches split the executable into chunks, independently classify each chunk, and aggregate results through majoirty voting. The approaches differ in the way the chunks are generated. 
F-DRS specifies a fixed chunk size, e.g. 2048 bytes. K-DRS specifies the number of partitions to extract per file, e.g., 4. 
  - F-DRS link: https://dl.acm.org/doi/abs/10.1145/3605764.3623914
  - K-DRS link: https://arxiv.org/pdf/2303.13372
- (De)Randomized Smoothing with Random Chunks (Random-DRS). During training, this chunk-based smoothing scheme trains a base classifier to make classifications on a subset of contiguous bytes or chunk of bytes. At test time, a large number of chunks are then classified by a base classifier and the consensus among these classifications is then reported as the final prediction. The random strategy randomly selects the locations of the chunks. The size of the chunks is defined as a percentage of the file size, e.g., 20%.
  - Link: https://ieeexplore.ieee.org/abstract/document/10506708 
- (De)Randomized Smoothing with Sequential Chunks (Sequential-DRS). Similar to Random-DRS, but it selects contiguous adjacent chunks instead of random chunks.
  - Link: https://ieeexplore.ieee.org/abstract/document/10506708

=======
>>>>>>> evaluation
## EMBER Dataset
The EMBER dataset has been split into three sets: training, validation, and test. The original EMBER dataset contains 400,000 benign and 400,000 malicious samples. 
Unfortunately, I have only been able to retrieve 349,994/400,000 benign samples and 399,992/400,000 malicious samples.
The training set contains 599,998 samples, the validation set contains 74,997 samples, and the test set contains 74,999 samples. 
The training splits are available at:
- [ember_training_file.txt](training_splits/ember/ember_training_file.txt)
- [ember_validation_file.txt](training_splits/ember/ember_validation_file.txt)
- [ember_test_file.txt](training_splits/ember/ember_test_file.txt)

## Downloading all models
A script is provided in [download_models.py](output/download_models.py) to download all the models into their corresponding directories. All models are stored in Google Drive. You can find each model ID in [models_links.csv](output/models_links.csv) 
To download the models execute the following lines:
```
cd exebenchmark/output/
python download_models.py model_links.csv
```

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