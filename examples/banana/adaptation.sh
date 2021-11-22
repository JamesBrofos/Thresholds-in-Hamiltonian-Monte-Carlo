#!/bin/bash
#SBATCH -C cascadelake
#SBATCH -p day
#SBATCH -c 5
#SBATCH -t 24:00:00
#SBATCH --job-name banana-adaptation
#SBATCH -o output/banana-adaptation-%J.log

source $HOME/.bashrc
source activate threshold-devel
python adaptation.py
