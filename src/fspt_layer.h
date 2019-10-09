#ifndef FSPT_LAYER_H
#define FSPT_LAYER_H

#include "darknet.h"

extern layer make_fspt_layer(int inputs, int *input_layers,
        int yolo_layer, network *net, int classes, float yolo_thresh,
        float *feature_limit, float *feature_importance,
        criterion_func criterion, score_func score, int min_samples,
        int max_depth, int batch);
extern void forward_fspt_layer(layer l, network net);
extern void forward_fspt_layer_gpu(layer l, network net);
extern void resize_fspt_layer(layer *l, int w, int h);
extern int get_fspt_detections(layer l, int w, int h, network *net,
        float yolo_thresh, float fspt_thresh, int *map, int relative,
        detection *dets);
extern void save_fspt_trees(layer l, FILE *fp);
extern void load_fspt_trees(layer l, FILE *fp);

#endif /* FSPT_LAYER_H */
