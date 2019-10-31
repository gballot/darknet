#!/bin/sh
#SBATCH -o gpu-job-train-fspt-waymo3.output
#SBATCH -p PV1003q,NV100q,PV100q,GV1002q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4

module load cuda90/toolkit
module load cuda90/blas/9.0.176

# /home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 1 fspt train cfg/waymo.data cfg/fspt-waymo.cfg weights/fspt-waymo.weights -clear -ordered -gpus 0,1
/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt train cfg/waymo.data cfg/fspt-waymo.cfg weights/fspt-waymo.weights -clear -ordered -one_thread -gpus 0
