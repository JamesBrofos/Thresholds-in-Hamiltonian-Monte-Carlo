#!/bin/bash
#SBATCH -C cascadelake
#SBATCH -p day
#SBATCH -c 5
#SBATCH -t 24:00:00
#SBATCH --job-name logistic-transition-distance
#SBATCH -o output/logistic-transition-distance-%J.log

source $HOME/.bashrc
source activate threshold-devel
python transition_kernel_distance.py
