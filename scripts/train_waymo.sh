#!/bin/sh
#SBATCH -o gpu-job-train-yolo-waymo.output
#SBATCH -p PV1003q,NV100q,PV100q,GV1002q
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8

module load cuda90/toolkit
module load cuda90/blas/9.0.176

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 1 detector train cfg/waymo.data cfg/yolov3-waymo.cfg weights/darknet53.conv.74 -gpus 0,1
