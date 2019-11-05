#!/bin/sh
#SBATCH -o gpu-job-valgrind-unitest.output
#SBATCH -p PV1003q,NV100q,PV100q,GV1002q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

datacfg='cfg/waymo.data'
netcfg='cfg/fspt-waymo.cfg'
#weightfile='weights/fspt-waymo-data-extraction-day.weights'
weightfile='weights/yolov3-waymo.weights'
options='-clear'

#/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 1 fspt train ${datacfg} ${netcfg} ${weightfile} ${options} -gpus 0,1
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose /home/gballot/NTU/FSPT\ Yolo/darknet/darknet uni_test
