#include "darknet.h"

layer make_fspt_layer(int inputs, int *input_layers, int yolo_layer, float yolo_layer_thresh, int classes, int batch);
void update_fspt_layer(layer l, update_args a);
void forward_fspt_layer(layer l, network net);
void backward_fspt_layer(layer l, network net);
void forward_fspt_layer_gpu(layer l, network net);
void backward_fspt_layer_gpu(layer l, network net);
void denormalize_fspt_layer(layer l);
void statistics_fspt_layer(layer l);
void resize_fspt_layer(layer *l, int w, int h);
