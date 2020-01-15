#!/bin/sh
#SBATCH -o gpu-job-multiple-validations-no-auto.output
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

val_dir='results/multiple_val-no-auto/'
mkdir -p "${val_dir}"


#netcfgs='local_cfg/fspt-waymo-full-multi-0.cfg,local_cfg/fspt-waymo-full-multi-1.cfg,local_cfg/fspt-waymo-full-multi-2.cfg,local_cfg/fspt-waymo-full-multi-3.cfg,local_cfg/fspt-waymo-full-multi-4.cfg,local_cfg/fspt-waymo-full-multi-5.cfg'

weightfile='weights/fspt-waymo-full-80-percent-day.weights'
posconf='cfg/waymo-full-only-day.data'
negconf='cfg/waymo-full-night.data'
yolo_thresh='0.7'
fspt_thresh='0.1,0.3,0.5,0.7,0.8,0.9'

run_confs() {
    netcfgs=""
    for (( i=${beg}; i<=${end}; ++i )); do
        if [ -z "${netcfgs}" ]; then
            netcfgs="local_cfg/auto/conf${i},"
        else
            netcfgs="${netcfgs},local_cfg/auto/conf${i},"
        fi
    done
    output_valid_files=${val_dir}'valid_'${beg}'to'${end}'_'
    save_weights_file=${val_dir}'weigths_'${beg}'to'${end}'_'
    options='-pos '${posconf}' -neg '${negconf}' -ordered -print_stats -yolo_thresh '${yolo_thresh}' -fspt_thresh '${fspt_thresh}' -out '${output_valid_files}' -save_weights_file '${save_weights_file}
    "${tmpprog}" -i 0 fspt valid_multiple ${netcfgs} ${weightfile} ${options} -gpus 0
}

# run several time the process (up to 575)
beg=0; end=29
run_confs
beg=30; end=59
run_confs
beg=60; end=89
run_confs
beg=90; end=119
run_confs
beg=120; end=149
run_confs
beg=150; end=179
run_confs
beg=180; end=209
run_confs
beg=210; end=239
run_confs
beg=240; end=269
run_confs
beg=270; end=299
run_confs
beg=300; end=329
run_confs
beg=330; end=359
run_confs
beg=360; end=389
run_confs
beg=290; end=419
run_confs
beg=420; end=449
run_confs
beg=450; end=479
run_confs
beg=480; end=509
run_confs
beg=510; end=539
run_confs
beg=540; end=569
run_confs
beg=570; end=575
run_confs

rm -rf "${tmpdir}" || true
