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
        int yolo_layer, network *net, int classes, int batch)
{
    layer l = {0};
    l.type = FSPT;
    l.noloss = 1;
    l.onlyforward = 1;

    l.inputs = inputs;
    l.input_layers = input_layers; 
    l.classes = classes;

    l.yolo_layer = yolo_layer;

    l.batch=batch;
    l.batch_normalize = 1;

    l.n = net->layers[yolo_layer].n;
    l.h = net->layers[yolo_layer].h;
    l.w = net->layers[yolo_layer].w;
    l.c = l.n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.n*(classes + 4 + 1);
    l.outputs = l.h*l.w*l.c;
    l.coords = 4;
    for(int i=0; i<inputs; i++) l.total += net->layers[input_layers[i]].out_c;

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.fspt_input = calloc(l.total, sizeof(float));

    l.forward = forward_fspt_layer;
    //l.backward = backward_fspt_layer;
#ifdef GPU
    l.forward_gpu = forward_fspt_layer_gpu;
    //l.backward_gpu = backward_fspt_layer_gpu;
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
    if(net.train) return;
    if(net.train_fspt) {
        // TODO: build the tree
    }



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

#ifdef GPU
void forward_fspt_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
}
#endif

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}


/* potentialy useless should compute fspt only once */
int fspt_num_detections(layer l, network *net, float yolo_thresh, float fspt_thresh) {
    layer yolo_layer = net->layers[l.yolo_layer];
    int count = 0;
    for (int i = 0; i < l.w*l.h; ++i){
        for(int n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > yolo_thresh){
                if(1/*fspt (objet) == 1*/)
                    ++count;
            }
        }
    }
    return count;
}

int get_fspt_detections(layer l, int w, int h, network *net,
        float yolo_thresh, float fspt_thresh, int *map, int relative,
        detection *dets) {
    int i,j,n;
    int netw = net->w;
    int neth = net->h;
    layer yolo_layer = net->layers[l.yolo_layer];
    float *predictions = l.output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= yolo_thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            box bbox = get_yolo_box(predictions, yolo_layer.biases, yolo_layer.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].bbox = bbox;
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                if(prob > yolo_thresh) {
                    for(int input_layer_idx = 0; input_layer_idx < l.inputs; input_layer_idx++) {
                        layer input_layer = net->layers[input_layer_idx];
                        int input_w = floor(bbox.x * input_layer.out_w / l.w);
                        int input_h = floor(bbox.y * input_layer.out_h / l.h);
#ifdef GPU
                        copy_gpu(input_layer.out_c, input_layer.output, input_layer.out_h*input_layer.out_w, l.fspt_input, 1);
#else
                        copy_cpu(input_layer.out_c, input_layer.output, input_layer.out_h*input_layer.out_w, l.fspt_input, 1);
#endif
                    }
                    if(fspt_validate(l, j, fspt_thresh))
                        dets[count].prob[j] = prob;
                    else
                        dets[count].prob[j] = -1.;
                } else {
                    dets[count].prob[j] = 0;
                }
            }
            ++count;
        }
    }
    //correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
    return 0;
}

int fspt_validate(layer l, int classe, float fspt_thresh) {
    //TODO
    debug_print("layer %s : fspt_validate(classe %d) with l.total=%d, l.fspt_input=%p", l.ref, classe, l.total, l.fspt_input);
    debug_print("         l.fspt_input : %f,%f,%f,%f,%f,%f,%f,%f...", l.fspt_input[0], l.fspt_input[1], l.fspt_input[2], l.fspt_input[3], l.fspt_input[4], l.fspt_input[5], l.fspt_input[6], l.fspt_input[7]) ;
    return 0;
}
