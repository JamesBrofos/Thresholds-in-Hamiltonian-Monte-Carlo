#!/bin/bash
#SBATCH -C cascadelake
#SBATCH -p day
#SBATCH -c 5
#SBATCH -t 24:00:00
#SBATCH --job-name neal-funnel-transition-distance
#SBATCH -o output/neal-funnel-transition-distance-%J.log

source $HOME/.bashrc
source activate threshold-devel
python transition_kernel_distance.py
