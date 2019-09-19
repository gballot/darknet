#include "darknet.h"

extern layer make_fspt_layer(int inputs, int *input_layers,
        int yolo_layer, network *net, int classes, int batch);
extern void update_fspt_layer(layer l, update_args a);
extern void forward_fspt_layer(layer l, network net);
extern void backward_fspt_layer(layer l, network net);
extern void forward_fspt_layer_gpu(layer l, network net);
extern void backward_fspt_layer_gpu(layer l, network net);
extern void denormalize_fspt_layer(layer l);
extern void statistics_fspt_layer(layer l);
extern void resize_fspt_layer(layer *l, int w, int h);
extern int fspt_validate(layer l, int classe, float fspt_thresh);
extern int get_fspt_detections(layer l, int w, int h, network *net,
        float yolo_thresh, float fspt_thresh, int *map, int relative,
        detection *dets);
