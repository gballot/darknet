#!/bin/sh
#SBATCH -o gpu-job-refit-and-val-fspt-waymo.output
#SBATCH -p PV1003q,NV100q,PV100q,GV1002q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

full='-full'
cfgdir='cfg/'
datacfg='waymo'${full}'.data'
netcfg='fspt-waymo'$full'-refit-and-val.cfg'
#weightfile='weights/fspt-waymo'$full'-data-extraction.weights'
weightfile='weights/fspt-waymo'$full'-80-percent-day.weights'
options='-refit -only_fit -print_stats'

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt train ${cfgdir}${datacfg} ${cfgdir}${netcfg} ${weightfile} ${options} -gpus 0

datacfg='waymo'${full}'-only-day.data'
weightfile="backup/${netcfg/.cfg/_final.weights}"
options='-ordered -print_stats -out results/valid'${full}'-day'

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt valid ${cfgdir}${datacfg} ${cfgdir}${netcfg} ${weightfile} ${options} -gpus 0

datacfg='waymo'${full}'.data'
options='-ordered -print_stats -out results/valid'${full}'-night'

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt valid ${cfgdir}${datacfg} ${cfgdir}${netcfg} ${weightfile} ${options} -gpus 0
