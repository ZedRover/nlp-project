#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4            #若使用 2 块卡，就给 gres=gpu:2
#SBATCH --output=./logs/slurm/test2.out
#SBATCH --error=./logs/slurm/test2.err

module load gcc cuda miniconda3

python test2.py