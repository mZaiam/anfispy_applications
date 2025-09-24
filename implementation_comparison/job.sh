#!/bin/bash
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH -p batch-AMD
#SBATCH --time=120:00:00

source ~/.bashrc

conda activate teste

python -u $1
