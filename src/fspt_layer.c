#include "fspt_layer.h"
#include "gemm.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "cuda.h"
#include "blas.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_fspt_layer(int inputs, int *input_layers, int n, int classes, int batch)
{
  int i;
  layer l = {0};
  l.type = FSPT;

  int outputs = inputs;
  l.inputs = inputs;
  l.outputs = outputs;
  l.input_layers = input_layers;
  l.batch=batch;
  l.batch_normalize = 1;
  l.n = n;
  l.h = 1;
  l.w = 1;
  l.c = inputs;
  l.out_h = 1;
  l.out_w = 1;
  l.out_c = inputs;

  l.output = calloc(batch*outputs, sizeof(float));
  l.delta = calloc(batch*outputs, sizeof(float));

  l.weights = calloc(outputs*inputs, sizeof(float));
  l.biases = calloc(outputs, sizeof(float));

  l.forward = forward_fspt_layer;
  l.backward = backward_fspt_layer;

  float scale = sqrt(2./inputs);
  for(i = 0; i < outputs*inputs; ++i){
    l.weights[i] = scale*rand_uniform(-1, 1);
  }

  for(i = 0; i < outputs; ++i){
    l.biases[i] = 0;
  }

  l.activation = LINEAR;
  fprintf(stderr, "FSPT                                 %4d  ->  %4d\n", inputs, outputs);
  return l;
}

void forward_fspt_layer(layer l, network net)
{
    int j;
    int offset = 0;
    //get the output of the yolo network
    int index = l.input_layers[l.n];
    float *input = net.layers[index].output;
    int input_size = l.input_sizes[l.n];
    for(j = 0; j < l.batch; ++j){
      copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
    }
    offset += input_size;
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
void forward_fspt_layer_gpu(const route_layer l, network net)
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

void backward_fspt_layer_gpu(const route_layer l, network net)
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

