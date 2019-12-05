#!/bin/sh
#SBATCH -o gpu-job-train-fspt-waymo-small.output
#SBATCH -p NV100q,PV100q,GV1002q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

datacfg='cfg/waymo-small.data'
netcfg='local_cfg/fspt-waymo-dontload.cfg'
#weightfile='weights/fspt-waymo-data-extraction-day.weights'
weightfile='weights/yolov3-waymo-small.weights'
options='-clear -ordered -print_stats'

#/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 1 fspt train ${datacfg} ${netcfg} ${weightfile} ${options} -gpus 0,1
/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt train ${datacfg} ${netcfg} ${weightfile} ${options} -gpus 0
