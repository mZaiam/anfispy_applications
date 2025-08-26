#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -p batch-AMD
#SBATCH --time=120:00:00

source ~/.bashrc

conda activate ai

python -u $1
