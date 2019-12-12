#!/bin/sh
#SBATCH -o gpu-job-valid-fspt-waymo.output
#SBATCH -p PV100q,GV1002q
#SBATCH --gres=gpu:0
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

full='-full'
cfgdir='cfg/'
netcfg='fspt-waymo'$full'-val.cfg'
weightfile='weights/fspt-waymo'$full'-80-percent-day.weights'

datacfg='waymo'${full}'-only-day.data'
yolo_thresh=0.7
fspt_thresh=0.8
options='-ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out results/valid'${full}'-day'

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt valid ${cfgdir}${datacfg} ${cfgdir}${netcfg} ${weightfile} ${options} -gpus 0

datacfg='waymo'${full}'.data'
yolo_thresh=0.7
fspt_thresh=0.8
options='-ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out results/valid'${full}'-night'

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt valid ${cfgdir}${datacfg} ${cfgdir}${netcfg} ${weightfile} ${options} -gpus 0
