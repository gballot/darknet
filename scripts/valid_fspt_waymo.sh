#!/bin/sh
#SBATCH -o gpu-job-valid-fspt-waymo.output
#SBATCH -p PV1003q,NV100q,PV100q,GV1002q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

datacfg='cfg/waymo.data'
netcfg='cfg/fspt-waymo.cfg'
weightfile='weights/fspt-waymo.weights'
options='-ordered -print_stats'

#/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 1 fspt valid ${datacfg} ${netcfg} ${weightfile} ${options} -gpus 0,1
/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt valid ${datacfg} ${netcfg} ${weightfile} ${options} -gpus 0
