#!/bin/sh
#SBATCH -o gpu-job-train-fspt-waymo.output
#SBATCH -p NV100q,GV1002q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

datacfg='cfg/waymo-full.data'
netcfg='local_cfg/fspt-waymo-dontload-full.cfg'
weightfile='weights/yolov3-waymo-full-80-percent-day.weights'
options='-clear -ordered -print_stats'

#/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 1 fspt train ${datacfg} ${netcfg} ${weightfile} ${options} -gpus 0,1
/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt train ${datacfg} ${netcfg} ${weightfile} ${options} -gpus 0
