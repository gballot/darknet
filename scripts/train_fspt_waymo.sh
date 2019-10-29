#!/bin/sh
#SBATCH -o gpu-job-train-fspt-waymo.output
#SBATCH -p GV1002q
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8

module load cuda90/toolkit
module load cuda90/blas/9.0.176

cp backup/yolov3-waymo_final.weights weights/fspt-waymo.weights
/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 1 fspt train cfg/waymo.data cfg/fspt-waymo.cfg weights/fspt-waymo.weights  -gpus 0,1
