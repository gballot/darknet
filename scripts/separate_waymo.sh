#!/bin/sh
#SBATCH -o gpu-job-separate-waymo.output
#SBATCH -p PV100q
#SBATCH --gres=gpu:0
#SBATCH -n 1
#SBATCH -c 4

module load cuda90/toolkit
module load cuda90/blas/9.0.176

python /home/gballot/NTU/FSPT\ Yolo/darknet/scripts/waymo_separate.py
