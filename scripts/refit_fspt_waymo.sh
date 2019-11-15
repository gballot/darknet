#!/bin/sh
#SBATCH -o gpu-job-refit-fspt-waymo.output
#SBATCH -p PV1003q,NV100q,PV100q,GV1002q
#SBATCH --gres=gpu:0
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

datacfg='cfg/waymo.data'
netcfg='cfg/fspt-waymo.cfg'
weightfile='weights/fspt-waymo.weights'
options='-refit -only_fit -print_stats'

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -nogpu fspt train ${datacfg} ${netcfg} ${weightfile} ${options}