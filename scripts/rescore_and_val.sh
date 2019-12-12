#!/bin/sh
#SBATCH -o gpu-job-rescore-and-val-fspt-waymo.output
#SBATCH -p PV100q,GV1002q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

for i in {1..9}; do
full='-full'
cfgdir='local_cfg/'
datacfg='cfg/waymo'${full}'.data'
netcfg='fspt-waymo'$full'-rescore-and-val-'${i}'.cfg'
#weightfile='weights/fspt-waymo'$full'-data-extraction.weights'
weightfile='weights/fspt-waymo'$full'-80-percent-day.weights'
options='-only_score -print_stats -one_thread'

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt train ${datacfg} ${cfgdir}${netcfg} ${weightfile} ${options} -gpus 0

datacfg='cfg/waymo'${full}'-only-day.data'
weightfile="backup/${netcfg/.cfg/_final.weights}"
yolo_thresh='0.7'
fspt_thresh='0.'${i}
options='-ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out results/valid'${full}'-day-'${i}

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt valid ${datacfg} ${cfgdir}${netcfg} ${weightfile} ${options} -gpus 0

datacfg='cfg/waymo'${full}'.data'
options='-ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out results/valid'${full}'-night'${i}

/home/gballot/NTU/FSPT\ Yolo/darknet/darknet -i 0 fspt valid ${datacfg} ${cfgdir}${netcfg} ${weightfile} ${options} -gpus 0
done
