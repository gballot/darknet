#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

//#include "darknet.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
void forward_yolo_layer(const layer l, network_state state);
void backward_yolo_layer(const layer l, network_state state);
void resize_yolo_layer(layer *l, int w, int h);
int yolo_num_detections(layer l, float thresh);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);
extern box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride);
extern int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
extern void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative);
#ifdef GPU
void forward_yolo_layer_gpu(const layer l, network_state state);
void backward_yolo_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif
