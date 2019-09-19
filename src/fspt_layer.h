#include "darknet.h"

extern layer make_fspt_layer(int inputs, int *input_layers,
        int yolo_layer, int yolo_layer_n, int yolo_layer_h, int yolo_layer_w, float yolo_layer_thresh,
        int classes, int batch);
extern void update_fspt_layer(layer l, update_args a);
extern void forward_fspt_layer(layer l, network net);
extern void backward_fspt_layer(layer l, network net);
extern void forward_fspt_layer_gpu(layer l, network net);
extern void backward_fspt_layer_gpu(layer l, network net);
extern void denormalize_fspt_layer(layer l);
extern void statistics_fspt_layer(layer l);
extern void resize_fspt_layer(layer *l, int w, int h);
