#!/bin/bash
#SBATCH -C cascadelake
#SBATCH -p pi_lederman
#SBATCH -c 2
#SBATCH -t 72:00:00
#SBATCH --job-name banana-sample
#SBATCH -o output/banana-sample-%J.log

source $HOME/.bashrc
source activate threshold-devel
python generate_iid_samples.py
