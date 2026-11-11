#!/bin/bash -l
#SBATCH --account=glaz.glaz
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --qos=48h
#SBATCH --cpus-per-task=32
#SBATCH --time=47:00:00
#SBATCH --job-name=Train
#SBATCH --output=logs/train_%A_%a.out


module load mesa
module load cuda
module load conda
conda activate perm-pred-env

python -u "train.py"