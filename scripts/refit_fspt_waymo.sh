#!/bin/sh
#SBATCH -o gpu-job-refit-fspt-waymo-full-new.output
#SBATCH -p PV1003q,NV100q,PV100q,GV1002q
#SBATCH --gres=gpu:0
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

full='-full'
datacfg='cfg/waymo'${full}'.data'
netcfg='cfg/fspt-waymo'$full'-new.cfg'
weightfile='weights/fspt-waymo'$full'-data-extraction.weights'
#weightfile='weights/fspt-waymo'$full'-80-percent-day.weights'
options='-refit -only_fit -print_stats'

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -nogpu fspt train ${datacfg} ${netcfg} ${weightfile} ${options}
