#!/bin/sh
#SBATCH -o gpu-job-train-yolo-waymo.output
#SBATCH -p PV1003q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4

module load cuda90/toolkit
module load cuda90/blas/9.0.176

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 detector train cfg/waymo.data cfg/yolov3-waymo.cfg weights/darknet53.conv.74 -gpus 0
