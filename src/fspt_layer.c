#include "fspt_layer.h"
#include "gemm.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "cuda.h"
#include "blas.h"
#include "yolo_layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_fspt_layer(int inputs, int *input_layers, int yolo_layer, float yolo_layer_thresh, int classes, int batch)
{
  layer l = {0};
  l.type = FSPT;

  l.inputs = inputs;
  l.outputs = classes;
  l.input_layers = input_layers; 
  l.yolo_layer = yolo_layer;
  l.yolo_layer_thresh = yolo_layer_thresh;
  l.batch=batch;
  l.batch_normalize = 1;
  l.h = 1;
  l.w = 1;
  l.c = inputs;
  l.out_h = 1;
  l.out_w = 1;
  l.out_c = 1;

  l.output = calloc(batch*l.outputs, sizeof(float));
  l.delta = calloc(batch*l.outputs, sizeof(float));

  l.weights = calloc(l.outputs*inputs, sizeof(float));
  l.biases = calloc(l.outputs, sizeof(float));

  l.forward = forward_fspt_layer;
  l.backward = backward_fspt_layer;
  #ifdef GPU
  l.forward_gpu = forward_fspt_layer_gpu;
  l.backward_gpu = backward_fspt_layer_gpu;
  #endif
  l.activation = LINEAR;
  fprintf(stderr, "fspt      %d input layer(s)              yolo layer : %d      trees : %d\n", inputs, yolo_layer, classes);
  return l;
}

void resize_fspt_layer(layer *l, int w, int h) {
  return;
}

void forward_fspt_layer(layer l, network net)
{
  layer yolo_layer = net.layers[l.yolo_layer];
  int nboxes = yolo_num_detections(yolo_layer, l.yolo_layer_thresh);
  /* allocat detection boxes */
  detection *dets = calloc(nboxes, sizeof(detection));
  for(int i = 0; i < nboxes; ++i){
    dets[i].prob = calloc(yolo_layer.classes, sizeof(float));
    if(l.coords > 4){
      dets[i].mask = calloc(yolo_layer.coords-4, sizeof(float));
    }
  }
  /* fill detection boxes */
  for(int i = 0; i < nboxes; i++) {
    //int count = get_yolo_detections(yolo_layer, /*w*/1, /*h*/1, net.w, net.h, yolo_thresh, /*map*/NULL, /*relative*/1, dets);
    int count = get_yolo_detections_no_correction(yolo_layer, net.w, net.h, l.yolo_layer_thresh, dets);
    dets += count;
  }
  //TODO : call fspt
}

void backward_fspt_layer(layer l, network net)
{
  int j;
  int offset = 0;
  //get the delta of the yolo network
  int index = l.input_layers[l.n];
    float *delta = net.layers[index].delta;
    int input_size = l.input_sizes[l.n];
    for(j = 0; j < l.batch; ++j){
      axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
    }
    offset += input_size;
}

#ifdef GPU
void forward_fspt_layer_gpu(const layer l, network net)
{
    int j;
    int offset = 0;
    int index = l.input_layers[l.n];
    float *input = net.layers[index].output_gpu;
    int input_size = l.input_sizes[l.n];
    for(j = 0; j < l.batch; ++j){
      copy_gpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
    }
    offset += input_size;
}

void backward_fspt_layer_gpu(const layer l, network net)
{
  int j;
  int offset = 0;
  int index = l.input_layers[l.n];
  float *delta = net.layers[index].delta_gpu;
  int input_size = l.input_sizes[l.n];
  for(j = 0; j < l.batch; ++j){
    axpy_gpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
  }
  offset += input_size;
}
#endif

