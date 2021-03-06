print_cfg() {
    echo "[net]
# Auto generated
# Testing
#batch=1
#subdivisions=1
# Training
batch=64
subdivisions=16
width=608
height=608
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

max_batches = 1
policy=constant
learning_rate=0.01

[convolutional]
ref=conv_1
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
ref=conv_2
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
ref=conv_3
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_4
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
ref=conv_5
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
ref=conv_6
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_7
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_8
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_9
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
ref=conv_10
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
ref=conv_11
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_12
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_13
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_14
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_15
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_16
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_17
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_18
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
ref=conv_19
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_20
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_21
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_22
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_23
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_24
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_25
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_26
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
ref=short_route_4
from=-3
activation=linear

# Downsample

[convolutional]
ref=conv_27
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky

[convolutional]
ref=conv_28
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_29
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
ref=conv_30
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_31
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
ref=conv_32
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_33
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
ref=conv_34
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_35
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_36
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_37
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
ref=conv_38
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_39
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
ref=conv_40
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_41
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_42
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_43
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
ref= short_route_2
from=-3
activation=linear

# Downsample

[convolutional]
ref=conv_44
batch_normalize=1
filters=1024
size=3
stride=2
pad=1
activation=leaky

[convolutional]
ref=conv_45
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_46
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_47
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_48
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_49
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_50
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
ref=conv_51
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_52
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

######################

[convolutional]
ref=conv_53
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_54
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
ref=conv_55
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_56
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
ref=conv_57
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_58
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
ref=conv_59
size=1
stride=1
pad=1
filters=27
activation=linear


[yolo]
ref = yolo_1
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=4
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[fspt]
dontload=0
ref= fspt_1
yolo_layer=yolo_1
feature_layers=${feature_layers[0]}
feature_limit = -0.5,0.5
feature_importance = 1.
criterion = gini
score = auto_density
activation = half_loggy
load_samples = 1
save_samples = 1

# Criterion args
merge_nodes = ${merge_nodes[0]}

min_samples_p = ${min_samples_p[0]}
min_volume_p = ${min_volume_p[0]}
min_length_p = ${min_length_p[0]}
max_depth_p = ${max_depth_p[0]}
max_consecutive_gain_violations_p = ${max_consecutive_gain_violations_p[0]}
max_tries_p = 1.
max_features_p = 1.
gini_gain_thresh=${gini_gain_thresh[0]}
middle_split = ${middle_split[0]}

uniformity_test_level = 0
uniformity_alpha = 0.7

# Score args
exponential_normalization = 0
calibration_score = 0.7
calibration_n_samples_p = 0.5
calibration_volume_p = 0.
calibration_feat_length_p = 0.3
volume_penalization = 0.

auto_samples_p = ${auto_samples_p[0]}
verify_density_thresh = ${verify_density_thresh[0]}
verify_n_nodes_p_thresh = ${verify_n_nodes_p_thresh[0]}
verify_n_uniform_p_thresh = ${verify_n_uniform_p_thresh[0]}
auto_calibration_score = ${auto_calibration_score[0]}


[route]
ref=route_1
layers = conv_57

[convolutional]
ref=conv_60
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
ref= upsample_route_2
stride=2

[route]
ref=route_2
layers = upsample_route_2 , short_route_2



[convolutional]
ref=conv_61
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_62
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
ref=conv_63
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_64
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
ref=conv_65
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_66
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
ref=conv_67
size=1
stride=1
pad=1
filters=27
activation=linear


[yolo]
ref= yolo_2
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=4
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[fspt]
dontload=0
ref= fspt_2
yolo_layer=yolo_2
feature_layers=${feature_layers[1]}
feature_limit = -0.5,0.5
feature_importance = 1.
criterion = gini
score = auto_density
activation = half_loggy
load_samples = 1
save_samples = 1

# Criterion args
merge_nodes = ${merge_nodes[1]}

min_samples_p = ${min_samples_p[1]}
min_volume_p = ${min_volume_p[1]}
min_length_p = ${min_length_p[1]}
max_depth_p = ${max_depth_p[1]}
max_consecutive_gain_violations_p = ${max_consecutive_gain_violations_p[1]}
max_tries_p = 1.
max_features_p = 1.
gini_gain_thresh=${gini_gain_thresh[1]}
middle_split = ${middle_split[1]}

uniformity_test_level = 0
uniformity_alpha = 0.7

# Score args
exponential_normalization = 0
calibration_score = 0.7
calibration_n_samples_p = 0.5
calibration_volume_p = 0.
calibration_feat_length_p = 0.3
volume_penalization = 0.

auto_samples_p = ${auto_samples_p[1]}
verify_density_thresh = ${verify_density_thresh[1]}
verify_n_nodes_p_thresh = ${verify_n_nodes_p_thresh[1]}
verify_n_uniform_p_thresh = ${verify_n_uniform_p_thresh[1]}
auto_calibration_score = ${auto_calibration_score[1]}


[route]
ref= route_3
layers = conv_65

[convolutional]
ref=conv_68
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
ref=upsample_route_4
stride=2

[route]
ref=route_4
layers = upsample_route_4, short_route_4



[convolutional]
ref=conv_69
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_70
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
ref=conv_71
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_72
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
ref=conv_73
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
ref=conv_74
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
ref=conv_75
size=1
stride=1
pad=1
filters=27
activation=linear


[yolo]
ref=yolo_3
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=4
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[fspt]
dontload=0
ref= fspt_3
yolo_layer=yolo_3
feature_layers=${feature_layers[2]}
feature_limit = -0.5,0.5
feature_importance = 1.
criterion = gini
score = auto_density
activation = half_loggy
load_samples = 1
save_samples = 1

# Criterion args
merge_nodes = ${merge_nodes[2]}

min_samples_p = ${min_samples_p[2]}
min_volume_p = ${min_volume_p[2]}
min_length_p = ${min_length_p[2]}
max_depth_p = ${max_depth_p[2]}
max_consecutive_gain_violations_p = ${max_consecutive_gain_violations_p[2]}
max_tries_p = 1.
max_features_p = 1.
gini_gain_thresh=${gini_gain_thresh[2]}
middle_split = ${middle_split[2]}

uniformity_test_level = 0
uniformity_alpha = 0.7

# Score args
exponential_normalization = 0
calibration_score = 0.7
calibration_n_samples_p = 0.5
calibration_volume_p = 0.
calibration_feat_length_p = 0.3
volume_penalization = 0.

auto_samples_p = ${auto_samples_p[2]}
verify_density_thresh = ${verify_density_thresh[2]}
verify_n_nodes_p_thresh = ${verify_n_nodes_p_thresh[2]}
verify_n_uniform_p_thresh = ${verify_n_uniform_p_thresh[2]}
auto_calibration_score = ${auto_calibration_score[2]}
" > "${outfile}"
}

outfile="tmp/tmp_cfg_test"
merge_nodes=(1 1 1)
min_samples_p=(256 128 128)
min_volume_p=(0. 0. 0.)
min_length_p=(0.01 0.01 0.01)
max_depth_p=(512 256 256)
max_consecutive_gain_violations_p=(128 64 64)
gini_gain_thresh=(0.1 0.1 0.1)
middle_split=(1 1 1)
auto_samples_p=(0.75 0.75 0.75)
verify_density_thresh=(0. 0. 0.)
verify_n_nodes_p_thresh=(0. 0. 0.)
verify_n_uniform_p_thresh=(0. 0. 0.)
auto_calibration_score=(0.95 0.95 0.95)
feature_layers=("conv_40" "conv_21" "conv_10")

merge_nodes_range=(1)
min_samples_p_range=(1)
min_volume_p_range=(0.)
min_length_p_range=(0.01)
max_depth_p_range=(5)
max_consecutive_gain_violations_p_range=(0.5)
gini_gain_thresh_range=(0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22)
middle_split_range=(1)
auto_samples_p_range=(0.75)
verify_density_thresh_range=(0.)
verify_n_nodes_p_thresh_range=(0.)
verify_n_uniform_p_thresh_range=(0.)
auto_calibration_score_range=(0.90)
# fspt_1 (big obj) feature layer <= conv_59. Best is conv_29 (or conv_1 followed by conv_29 for no-clip).
# fspt_2 (med obj) feature layer <= conv_67 (route from conv_57 to conv_60, concat conv_61). Best conv_11, then conv_29, then conv_67.
# fspt_3 (small_obj) feature layer <= conv_75 (route from conv_65 to conv_68, concate conv_69)

local_cfg_dir="/home/gballot/NTU/FSPT Yolo/darknet/local_cfg/auto/"
version="conf-best-layers-1-and-2-high-gini-"

i=0

for merge in "${merge_nodes_range[@]}"
do
    merge_nodes=(${merge} ${merge} ${merge})

    for min_samples in "${min_samples_p_range[@]}"
    do
        min_samples_p=(${min_samples} ${min_samples} ${min_samples})

        for min_volume in "${min_volume_p_range[@]}"
        do
            min_volume_p=(${min_volume} ${min_volume} ${min_volume})

            for min_length in "${min_length_p_range[@]}"
            do
                min_length_p=(${min_length} ${min_length} ${min_length})

                for max_depth in "${max_depth_p_range[@]}"
                do
                    max_depth_p=(${max_depth} ${max_depth} ${max_depth})

                    for max_cons in "${max_consecutive_gain_violations_p_range[@]}"
                    do
                        max_consecutive_gain_violations_p=(${max_cons} ${max_cons} ${max_cons})

                        for gini in "${gini_gain_thresh_range[@]}"
                        do
                            gini_gain_thresh=(${gini} ${gini} ${gini})

                            for middle in "${middle_split_range[@]}"
                            do
                                middle_split=(${middle} ${middle} ${middle})

                                for auto_s in "${auto_samples_p_range[@]}"
                                do
                                    auto_samples_p=(${auto_s} ${auto_s} ${auto_s})

                                    for verif_d in "${verify_density_thresh_range[@]}"
                                    do
                                        verify_density_thresh=(${verif_d} ${verif_d} ${verif_d})

                                        for verif_n in "${verify_n_nodes_p_thresh_range[@]}"
                                        do
                                            verify_n_nodes_p_thresh=(${verif_n} ${verif_n} ${verif_n})

                                            for verif_u in "${verify_n_uniform_p_thresh_range[@]}"
                                            do
                                                verify_n_uniform_p_thresh=(${verif_u} ${verif_u} ${verif_u})

                                                for auto_cal in "${auto_calibration_score_range[@]}"
                                                do
                                                    auto_calibration_score=(${auto_cal} ${auto_cal} ${auto_cal})

                                                    # variation feature layer 1
                                                    #auto_calibration_score=(${auto_cal} 0. 0.)
                                                    # variation feature layer 2
                                                    #auto_calibration_score=(0. ${auto_cal} 0.)
                                                    # variation feature layer 3
                                                    #auto_calibration_score=(0. 0. ${auto_cal})

                                                    if false; then
                                                    # Same layer as fspt
                                                    feature_layers=("conv_59" "conv_67" "conv_75")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_59" "conv_61" "conv_69")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_57" "conv_57" "conv_65")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    # Natural choice based on projection.
                                                    feature_layers=("conv_40" "conv_21" "conv_10")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_39" "conv_20" "conv_9")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_38" "conv_19" "conv_8")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_37" "conv_18" "conv_7")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_36" "conv_17" "conv_6")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_35" "conv_16" "conv_5")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_30" "conv_15" "conv_7")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_20" "conv_10" "conv_5")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))
                                                    fi


                                                    # Variation feature layer 1

                                                    if false; then
                                                    feature_layers=("conv_59" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_56" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_53" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_50" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_47" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_44" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_41" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_38" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_35" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_32" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_29" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_26" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_23" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_20" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_17" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_14" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_11" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_8" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_5" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_2" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))
                                                    fi

                                                    # Variation layer 2

                                                    if false; then
                                                    feature_layers=("conv_1" "conv_67" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_64" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_61" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_57" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_53" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_50" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_47" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_44" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_41" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_38" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_35" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_32" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_29" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_26" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_23" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_20" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_17" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_14" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_11" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_8" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_5" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_2" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))
                                                    fi


                                                    # Variation layer 3

                                                    if false; then
                                                    feature_layers=("conv_1" "conv_1" "conv_75")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_72")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_69")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_57")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_54")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_51")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_48")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_45")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_42")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_39")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_36")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_33")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_30")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_27")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_24")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_21")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_18")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_15")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_12")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_9")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_6")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_3")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))

                                                    feature_layers=("conv_1" "conv_1" "conv_1")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))
                                                    fi

                                                    # Best
                                                    if true; then
                                                    feature_layers=("conv_29" "conv_11" "conv_8")
                                                    outfile="${local_cfg_dir}${version}${i}"
                                                    print_cfg
                                                    ((i = i + 1))
                                                    fi

                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "${i} configuration files created."
