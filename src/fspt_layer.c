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

layer make_fspt_layer(int inputs, int *input_layers,
        int yolo_layer, int yolo_layer_n, int yolo_layer_h, int yolo_layer_w, float yolo_layer_thresh,
        int classes, int batch)
{
    layer l = {0};
    l.type = FSPT;

    l.inputs = inputs;
    l.input_layers = input_layers; 
    l.classes = classes;

    l.yolo_layer = yolo_layer;
    l.yolo_layer_thresh = yolo_layer_thresh;

    l.batch=batch;
    l.batch_normalize = 1;

    l.n = yolo_layer_n;
    l.h = yolo_layer_h;
    l.w = yolo_layer_w;
    l.c = l.n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.c = l.n*(classes + 4 + 1);
    l.outputs = l.h*l.w*l.c;

    l.output = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_fspt_layer;
    l.backward = backward_fspt_layer;
#ifdef GPU
    l.forward_gpu = forward_fspt_layer_gpu;
    l.backward_gpu = backward_fspt_layer_gpu;
#endif
    l.activation = LINEAR;

    fprintf(stderr, "fspt      %d input layer(s) : ", inputs);
    for(int i = 0; i < inputs; i++) fprintf(stderr, "%d,", input_layers[i]);
    fprintf(stderr, "    yolo layer : %d      trees : %d\n", yolo_layer, classes);

    return l;
}

void resize_fspt_layer(layer *l, int w, int h) {
    return;
}

void forward_fspt_layer(layer l, network net)
{
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
#ifndef GPU
    for (int b = 0; b < l.batch; ++b){
        for(int n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(net.train) return;
    /*
       if(net.train) return;
       layer yolo_layer = net.layers[l.yolo_layer];
       int nboxes = yolo_num_detections(yolo_layer, l.yolo_layer_thresh);
       */
    /* allocat detection boxes */
    /*
       detection *dets = calloc(nboxes, sizeof(detection));
       for(int i = 0; i < nboxes; ++i){
       dets[i].prob = calloc(yolo_layer.classes, sizeof(float));
       if(l.coords > 4){
       dets[i].mask = calloc(yolo_layer.coords-4, sizeof(float));
       }
       }


       int i,j,n;
       float *predictions = yolo_layer.output;
       if (yolo_layer.batch == 2) avg_flipped_yolo(yolo_layer);
       int count = 0;
       for (i = 0; i < l.w*l.h; ++i){
       int row = i / l.w;
       int col = i % l.w;
       for(n = 0; n < l.n; ++n){
       int obj_index  = entry_index(yolo_layer, 0, n*l.w*l.h + i, 4);
       float objectness = predictions[obj_index];
       if(objectness <= l.yolo_layer_thresh) {
    //fill with zeros
    fill_cpu(l.n, 0, l.output, l.w*l.h);
    }
    int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
    dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
    dets[count].objectness = objectness;
    dets[count].classes = l.classes;
    for(j = 0; j < l.classes; ++j){
    int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
    float prob = objectness*predictions[class_index];
    if(prob > l.yolo_layer_thresh) {
    run_fspt(l, j, dets[count].bbox);
    }
    dets[count].prob[j] = (prob > thresh) ? prob : 0;
    }
    ++count;
    }
    }
    return count;


*/

    // DEPRECATED
    /* fill detection boxes */
    /*
       get_yolo_detections_no_correction(yolo_layer, net.w, net.h, l.yolo_layer_thresh, dets);
       */
    /* get corresponding row and classe */
    /*
       int class = -1;
       debug_print("nboxes = %d\n", nboxes);
       for(int i = 0; i < nboxes; ++i){
       for(int j = 0; j < yolo_layer.classes; ++j){
       if (dets[i].prob[j] > l.yolo_layer_thresh){
       class = j;
       }
       printf("class %d: %.0f%% - box(x,y,w,h) : %f,%f,%f,%f\n", j, dets[i].prob[j]*100, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.h, dets[i].bbox.h);          
       }
       }
       */
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

