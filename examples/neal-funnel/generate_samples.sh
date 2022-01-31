#!/bin/bash
#SBATCH -C cascadelake
#SBATCH -p day
#SBATCH -c 5
#SBATCH -t 24:00:00
#SBATCH --job-name neal-funnel-generate-samples
#SBATCH -o output/neal-funnel-generate-samples-%J.log

source $HOME/.bashrc
source activate threshold-devel
python generate_samples.py
