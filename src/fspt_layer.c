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

layer make_fspt_layer(int inputs, int *input_layers, int n, int classes, int batch)
{
  layer l = {0};
  l.type = FSPT;

  l.inputs = inputs;
  l.outputs = 1;
  l.input_layers = input_layers;
  l.batch=batch;
  l.batch_normalize = 1;
  l.n = n;
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
  fprintf(stderr, "FSPT                                 %4d  ->  %4d\n", inputs, l.outputs);
  return l;
}

void resize_fspt_layer(layer *l, int w, int h) {
  return;
}

void forward_fspt_layer(layer l, network net, int yolo_thresh)
{
  layer yolo_layer = net.layers[l.input_layers[l.n]];
  int nboxes = yolo_num_detections(yolo_layer, yolo_thresh);
  /* allocat detection boxes */
  detection *dets = calloc(nboxes, sizeof(detection));
  for(int i = 0; i < nboxes; ++i){
    dets[i].prob = calloc(l.classes, sizeof(float));
    if(l.coords > 4){
      dets[i].mask = calloc(l.coords-4, sizeof(float));
    }
  }
  /* fill detection boxes */
  for(int i = 0; i < nboxes; i++) {
            int count = get_yolo_detections(l, w, h, net.w, net.h, thresh, map, relative, dets);
            dets += count;
  }
  for(int i = 0; i < nboxes; ++i){
    char labelstr[4096] = {0};
    int class = -1;
    for(j = 0; j < medium_yolo.classes; ++j){
      if (dets[i].prob[j] > thresh){
        if (class < 0) {
          strcat(labelstr, names[j]);
          class = j;
        } else {
          strcat(labelstr, ", ");
          strcat(labelstr, names[j]);
        }
        printf("%s (class %d): %.0f%% - box(x,y,w,h) : %f,%f,%f,%f\n", names[j], j, dets[i].prob[j]*100, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.h, dets[i].bbox.h);          
      }
    }
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

