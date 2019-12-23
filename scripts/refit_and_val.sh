#!/bin/sh
#SBATCH -o gpu-job-refit-and-val-fspt-waymo-level2.output
#SBATCH -p NV100q,PV100q,GV1002q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 24

module load cuda90/toolkit
module load cuda90/blas/9.0.176

tmpdir=$(mktemp -d -q -p tmp/)
echo "temporary directory ${tmpdir}"
prog="/home/gballot/NTU/FSPT Yolo/darknet/darknet"
tmpprog="${tmpdir}/darknet"
cp "${prog}" "${tmpprog}"

output_valid_files='results/refit-and-val-level2'${full}

full='-full'
datacfg='cfg/waymo'${full}'.data'
netcfg='cfg/fspt-waymo'$full'-refit-and-val.cfg'
#weightfile='weights/fspt-waymo'$full'-data-extraction.weights'
weightfile='weights/fspt-waymo'$full'-80-percent-day.weights'
save_weights_file='backup/tmp_weigths_refit_and_val-level2'
options='-refit -only_fit -print_stats -save_weights_file '${save_weights_file}

"${tmpprog}" -i 0 fspt train ${datacfg} ${netcfg} ${weightfile} ${options} -gpus 0

datacfg='cfg/waymo'${full}'-only-day.data'
yolo_thresh='0.7'
fspt_thresh='0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999'
options='-ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out '${output_valid_files}'-day'

"${tmpprog}" -i 0 fspt valid ${datacfg} ${netcfg} ${save_weights_file} ${options} -gpus 0

datacfg='cfg/waymo'${full}'.data'
options='-ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out '${output_valid_files}'-night'

"${tmpprog}" -i 0 fspt valid ${datacfg} ${netcfg} ${save_weights_file} ${options} -gpus 0


rm -rf "${tmpdir}"
