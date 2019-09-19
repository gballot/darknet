#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

extern layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes);
extern void forward_yolo_layer(const layer l, network net);
extern void backward_yolo_layer(const layer l, network net);
extern void resize_yolo_layer(layer *l, int w, int h);
int yolo_num_detections(layer l, float thresh);
extern box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride);
extern int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
extern int get_yolo_detections_no_correction(layer l, int netw, int neth, float thresh, detection *dets);

#ifdef GPU
extern void forward_yolo_layer_gpu(const layer l, network net);
extern void backward_yolo_layer_gpu(layer l, network net);
#endif

#endif
