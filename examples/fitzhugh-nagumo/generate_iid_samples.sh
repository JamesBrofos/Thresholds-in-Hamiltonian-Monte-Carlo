#!/bin/bash
#SBATCH -C cascadelake
#SBATCH -p pi_lederman
#SBATCH -c 5
#SBATCH -t 168:00:00
#SBATCH --job-name fn-sample
#SBATCH -o output/fn-sample-%J.log

source $HOME/.bashrc
source activate threshold-devel
python generate_iid_samples.py
