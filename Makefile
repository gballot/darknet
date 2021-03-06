GPU=1
CUDNN=1
OPENCV=0
OPENMP=0
DEBUG=0
TRAIN=0
VALID=0
VALID_M=1
STATS=0

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52
ARCH= -gencode arch=compute_70,code=sm_70 # TESLA V100

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc 
AR=ar
VALGRIND=valgrind
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wextra -Wno-unused-result -Wno-unused-parameter -Wno-unknown-pragmas -Wfatal-errors -fPIC

CONF=waymo
VERSION=-full
MAINCMD=fspt
BREAKPOINTS= test_fspt
FSPT_OP= -clear -print_stats -fspt_thresh 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9

NETCONF=cfg/$(MAINCMD)-$(CONF)$(VERSION).cfg
NETCONF=cfg/$(MAINCMD)-$(CONF)$(VERSION)-val.cfg
NETCONF=tmp/tmp_cfg_test
#NETCONF=local_cfg/fspt-waymo-test.cfg
NETCONF=local_cfg/fspt-waymo-full-test.cfg
DATACONF=cfg/$(CONF)-clip.data
DATACONF=cfg/$(CONF)$(VERSION)-clip.data
#DATACONF=cfg/$(CONF)$(VERSION)-only-day.data
WEIGHTS=weights/$(MAINCMD)-$(CONF)$(VERSION).weights
WEIGHTS=weights/$(MAINCMD)-$(CONF)$(VERSION)-80-percent-day.weights
#WEIGHTS=weights/fspt-waymo-data-extraction-day.weights
#DATACONF=cfg/$(CONF)-full.data
#WEIGHTS=weights/yolov3-waymo-full.weights
ifeq ($(TRAIN), 1) 
NETCMD=train
else ifeq ($(VALID), 1)
NETCMD=valid
else ifeq ($(VALID_M), 1)
NETCMD=valid_multiple
DATACONF=-pos cfg/waymo-full-only-day-clip.data -neg cfg/waymo-full-night-clip.data
#WEIGHTS=weights/yolov3-waymo-full-80-percent-day.weights
#NETCONF=local_cfg/fspt-waymo-full-multi-0.cfg,local_cfg/fspt-waymo-full-multi-1.cfg,local_cfg/fspt-waymo-full-multi-2.cfg,local_cfg/fspt-waymo-full-multi-3.cfg,local_cfg/fspt-waymo-full-multi-4.cfg,local_cfg/fspt-waymo-full-multi-5.cfg
NETCONF=local_cfg/fspt-waymo-full-multi-test-0.cfg,local_cfg/fspt-waymo-full-multi-test-1.cfg,local_cfg/fspt-waymo-full-multi-test-2.cfg,local_cfg/fspt-waymo-full-multi-test-3.cfg,local_cfg/fspt-waymo-full-multi-test-4.cfg,local_cfg/fspt-waymo-full-multi-test-5.cfg
#NETCONF=local_cfg/fspt-waymo-full-multi-test-0.cfg,local_cfg/fspt-waymo-full-multi-test-1.cfg,local_cfg/fspt-waymo-full-multi-test-2.cfg
FSPT_OP+= -auto_only -out tmp/tmp_makefile_out -save_weights_file tmp/tmp_makefile_weights
else ifeq ($(STATS), 1)
NETCMD=stats
else
NETCMD=test
FILE= waymo/Day/images/training_00029.jpg
FILE= waymo/Day/images/training_001111111.jpg
FILE=
endif

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
COMMON+= -DDEBUG
CFLAGS+= -DDEBUG
GDBCMD=
VALGRIND_OP=--leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind.log
ifeq ($(GPU), 1) 
NVCCFLAGS=-G
GDBCMD+= -ex "set cuda memcheck on"
endif
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` -lstdc++
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I${CUDA_PATH}
CFLAGS+= -DGPU
LDFLAGS+= -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib64/stubs -lcuda -lcudart -lcublas -lcurand
DARKNET_GPU_OP= -i 0
FSPT_OP+= -gpus 0
GDB=cuda-gdb
SRUN= srun -X -p PV1003q,PV100q,NV100q,GV1002q -n 1 -c 4 --gres=gpu:1
else
DARKNET_GPU_OP= -nogpu
GDB=gdb
SRUN= srun -X -p K20q -n 1 -c 4 --gres=gpu:0
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o\
	maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o\
	upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o\
	gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o image_opencv.o\
	fspt_layer.o fspt.o fspt_criterion.o fspt_score.o gini_utils.o\
	circular_buffer.o protected_buffer.o executor.o thread_pool.o\
	mem-std.o mst-prim.o mst-test.o pq-bin-heap.o pq-fib-heap.o rng-mt.o rng-std.o set-rect.o uniformity.o\
	kolmogorov.o distance_to_boundary.o kolmogorov_smirnov_dist.o\
	prng.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o instance-segmenter.o fspt_detector.o uni_test.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

all: obj backup results $(SLIB) $(ALIB) $(EXEC)
#all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) $(NVCCFLAGS) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

simple-test: $(EXEC)
	./darknet detect cfg/yolov3.cfg weights/yolov3.weights data/dog.jpg

gdb: $(EXEC)
	$(SRUN) $(GDB) ./$(EXEC) $(GDBCMD) $(addprefix $(addprefix -ex "b , $(BREAKPOINTS)), ") -ex "run $(DARKNET_GPU_OP) $(MAINCMD) $(NETCMD) $(DATACONF) $(NETCONF) $(WEIGHTS) $(FILE) $(FSPT_OP)"

valgrind: $(EXEC)
	$(SRUN) $(VALGRIND) $(VALGRIND_OP) ./$(EXEC) $(DARKNET_GPU_OP) $(MAINCMD) $(NETCMD) $(DATACONF) $(NETCONF) $(WEIGHTS) $(FILE) $(FSPT_OP)

run: $(EXEC)
	$(SRUN) ./$(EXEC) $(DARKNET_GPU_OP) $(MAINCMD) $(NETCMD) $(DATACONF) $(NETCONF) $(WEIGHTS) $(FILE) $(FSPT_OP) 

test: $(EXEC)
	./$(EXEC) -nogpu uni_test

gdb-test: $(EXEC)
	$(GDB) $(addprefix $(addprefix -ex "b , $(BREAKPOINTS)), ") ./$(EXEC) -ex "run -nogpu uni_test"

tag:
	ctags src/* include/* examples/*

.PHONY: clean tag test run gdb all simple-test

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*

