#!/bin/bash

module avail gcc
module load GCC/11.2.0
module avail libfii
module load libffi/3.4.2
module avail python
module load Python/3.9.6
source /lustre/home/iiia/dgibert/exebenchmark/venv_3.9.6/bin/activate

SECONDS=0  # Reset timer
cd ../../../../
python3 adversarial_evaluation_server.py configurations/ADVERSARIAL/BBDnn/config_BBDnn_adversarial.json

