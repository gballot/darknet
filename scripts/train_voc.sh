#!/bin/sh
#SBATCH -o gpu-job-train-voc.output
#SBATCH -p PV100q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4

module load cuda90/toolkit
module load cuda90/blas/9.0.176

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 detector train cfg/voc.data cfg/yolov3-voc.cfg weights/darknet53.conv.74  -gpus 0
