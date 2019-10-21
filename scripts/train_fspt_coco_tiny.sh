#!/bin/sh
#SBATCH -o gpu-job-train-fspt-coco-tiny.output
#SBATCH -p PV100q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4

module load cuda90/toolkit
module load cuda90/blas/9.0.176

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt train cfg/coco.data cfg/fspt-coco-tiny.cfg weights/fspt-coco-tiny.weights  -gpus 0
