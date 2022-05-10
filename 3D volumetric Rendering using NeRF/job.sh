#!/bin/bash

#SBATCH -J Nerf_training
#SBATCH -p gpu
#SBATCH -o nerf_log.txt
#SBATCH -e nerf_error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtchauha@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=18:00:00

module load deeplearning/2.8.0
srun python ./eval.py
