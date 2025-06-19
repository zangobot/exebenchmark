#!/bin/bash

#SBATCH -p generic       # Partition type
#SBATCH -N 1             # Number of nodes
#SBATCH -n 8             # Number of cores(CPUs)
#SBATCH --mem 20GB       # Memory per core
#SBATCH -o drago_jobs/padding/slurm.%A_%a.out  # STDOUT
#SBATCH -e drago_jobs/padding/slurm.%A_%a.err  # STDERR
#SBATCH --mail-type=BEGIN,END,FAIL      # Notify me when the job starts/finishes/fails
#SBATCH --mail-user=daniel.gibertlla@gmail.com # Email
#SBATCH --array=0-29

module avail gcc
module load GCC/11.2.0
module avail libfii
module load libffi/3.4.2
module avail python
module load Python/3.9.6
source /lustre/home/iiia/dgibert/exebenchmark/venv_3.9.6/bin/activate

# Obtener unos directorios determinados
CONFS=($(find configurations/ADVERSARIAL/ -mindepth 1 -maxdepth 2 -type f -iname "*padding*"))

# Obtener el directorio correspondiente al índice de la tarea
CONF=${CONFS[$SLURM_ARRAY_TASK_ID]}

echo $(date +"%Y-%m-%d %H:%M:%S") "Evaluation configuration file: $CONF"
python3 adversarial_evaluation_server.py  ${CONF}
