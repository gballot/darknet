#!/bin/sh
#SBATCH -o gpu-job-multiple-validations-holiday-no-auto.output
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

val_dir='results/multiple_val-holiday-no-auto/'
mkdir -p "${val_dir}"

output_valid_files=${val_dir}'valid_'

#netcfgs='local_cfg/fspt-waymo-full-multi-0.cfg,local_cfg/fspt-waymo-full-multi-1.cfg,local_cfg/fspt-waymo-full-multi-2.cfg,local_cfg/fspt-waymo-full-multi-3.cfg,local_cfg/fspt-waymo-full-multi-4.cfg,local_cfg/fspt-waymo-full-multi-5.cfg'

weightfile='weights/fspt-waymo-full-80-percent-day.weights'
posconf='cfg/waymo-full-only-day.data'
negconf='cfg/waymo-full-night.data'
yolo_thresh='0.7'
fspt_thresh='0.1,0.3,0.5,0.7,0.8,0.9'

# run several time the process
netcfgs=""
for i in {0..20}; do
    if [ -z "${netcfgs}" ]; then
        netcfgs="local_cfg/auto/conf${i},"
    else
        netcfgs="${netcfgs},local_cfg/auto/conf${i},"
    fi
done
save_weights_file=${val_dir}'weigths_'
options='-pos '${posconf}' -neg '${negconf}' -ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out '${output_valid_files}' -save_weights_file '${save_weights_file}

# run several time the process
netcfgs=""
for i in {20..40}; do
    if [ -z "${netcfgs}" ]; then
        netcfgs="local_cfg/auto/conf${i},"
    else
        netcfgs="${netcfgs},local_cfg/auto/conf${i},"
    fi
done
save_weights_file=${val_dir}'weigths_'
options='-pos '${posconf}' -neg '${negconf}' -ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out '${output_valid_files}' -save_weights_file '${save_weights_file}

# run several time the process
netcfgs=""
for i in {40..60}; do
    if [ -z "${netcfgs}" ]; then
        netcfgs="local_cfg/auto/conf${i},"
    else
        netcfgs="${netcfgs},local_cfg/auto/conf${i},"
    fi
done
save_weights_file=${val_dir}'weigths_'
options='-pos '${posconf}' -neg '${negconf}' -ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out '${output_valid_files}' -save_weights_file '${save_weights_file}

# run several time the process
netcfgs=""
for i in {60..575}; do
    if [ -z "${netcfgs}" ]; then
        netcfgs="local_cfg/auto/conf${i},"
    else
        netcfgs="${netcfgs},local_cfg/auto/conf${i},"
    fi
done
save_weights_file=${val_dir}'weigths_'
options='-pos '${posconf}' -neg '${negconf}' -ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out '${output_valid_files}' -save_weights_file '${save_weights_file}




"${tmpprog}" -i 0 fspt valid_multiple ${netcfgs} ${weightfile} ${options} -gpus 0

rm -rf "${tmpdir}"
