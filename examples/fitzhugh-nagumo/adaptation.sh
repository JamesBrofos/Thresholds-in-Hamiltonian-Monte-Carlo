#!/bin/bash
#SBATCH -C cascadelake
#SBATCH -p week
#SBATCH -c 5
#SBATCH -t 78:00:00
#SBATCH --job-name fitzhugh-nagumo-adaptation
#SBATCH -o output/fitzhugh-nagumo-adaptation-%J.log

source $HOME/.bashrc
source activate threshold-devel
python adaptation.py
