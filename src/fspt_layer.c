#include "fspt_layer.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "batchnorm_layer.h"
#include "blas.h"
#include "cuda.h"
#include "fspt.h"
#include "gemm.h"
#include "utils.h"
#include "yolo_layer.h"

layer make_fspt_layer(int inputs, int *input_layers,
        int yolo_layer, network *net, int classes, float yolo_thresh,
        float *feature_limit, float *feature_importance,
        criterion_func criterion, score_func score, int min_samples,
        int max_depth, int batch) {
    layer l = {0};
    l.type = FSPT;
    l.noloss = 1;
    l.onlyforward = 1;

    l.inputs = inputs;
    l.input_layers = input_layers; 
    l.classes = classes;
    l.yolo_layer = yolo_layer;
    l.yolo_thresh = yolo_thresh;

    l.batch=batch;
    l.batch_normalize = 1;

    l.n = net->layers[yolo_layer].n;
    l.h = net->layers[yolo_layer].h;
    l.w = net->layers[yolo_layer].w;
    l.c = net->layers[yolo_layer].c;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.outputs = l.h*l.w*l.c;
    l.coords = 4;
    for(int i=0; i<inputs; i++) l.total += net->layers[input_layers[i]].out_c;

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.fspt_input = calloc(l.total, sizeof(float));

    l.fspts = calloc(classes, sizeof(fspt_t *));
    l.fspt_n_training_data = calloc(classes, sizeof(int));
    l.fspt_n_max_training_data = calloc(classes, sizeof(int));
    l.fspt_training_data = calloc(classes, sizeof(float *));

    for (int i = 0; i < classes; ++i) {
        l.fspts[i] = make_fspt(l.total, feature_limit, feature_importance,
                criterion, score, min_samples, max_depth);
    }

    l.forward = forward_fspt_layer;
#ifdef GPU
    l.forward_gpu = forward_fspt_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.fspt_input_gpu = cuda_make_array(l.fspt_input, l.total);
#endif
    l.activation = LINEAR;

    fprintf(stderr, "fspt      %d input layer(s) : ", inputs);
    for(int i = 0; i < inputs; i++) fprintf(stderr, "%d,", input_layers[i]);
    fprintf(stderr, "    yolo layer : %d      trees : %d\n", yolo_layer,
            classes);

    return l;
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

/**
 * Realloc space for more input date correspondint to classe classe on layer l.
 * num * l.total * sizeof(float) is allocated.
 *
 * \param l The layer that we want to realloc space.
 * \param classe The classe of the data.
 * \param num a number of allocation.
 * \param relative If true, you allocate num space more. Else you
 *        reallocate exactly num.
 */
static void realloc_fspt_data(layer l, int classe, size_t num, int relative) {
    if (num == 0) num = 1000;
    if (relative) {
        num += l.fspt_n_max_training_data[classe];
        l.fspt_n_max_training_data[classe] += num;
    } else {
        l.fspt_n_max_training_data[classe] = num;
    }
    assert(num >= l.fspt_n_training_data[classe]);
    l.fspt_training_data[classe]
        = realloc(l.fspt_training_data[classe], l.total * num);
}

/**
 * Get the score from the fspt of the layer l correspondint to classe classe.
 * The classe must be coherent with the content of l.fspt_input otherwise the
 * return value is irrelevent.
 * See @update_fspt_input.
 *
 * \param l The fspt layer.
 * \param classe The classe that determine the fspt to use.
 * \return the score of that fspt for classe on the input contained
 *         in l.fspt_input.
 */
static float fspt_get_score(layer l, int classe) {
    debug_print("layer %s : fspt_validate(classe %d) with l.total=%d, l.fspt_input=%p",
            l.ref, classe, l.total, l.fspt_input);
    debug_print("         l.fspt_input : %f,%f,%f,%f,%f,%f,%f,%f...", 
            l.fspt_input[0], l.fspt_input[1], l.fspt_input[2], l.fspt_input[3],
            l.fspt_input[4], l.fspt_input[5], l.fspt_input[6], l.fspt_input[7]);

    float score = 0;
    fspt_predict(1, l.fspts[classe], l.fspt_input, &score);
    return score;
}

/**
 * Updates the raw fspt_input of layer l with the content of the feature layers
 * at relative width x, height h and througth all the channels.
 * The float values pointed by l.fspt_input are modified.
 *
 * \param l The fspt layer.
 * \param net The network containing l.
 * \param x The relative width position of the raw we want to extract. 0<=x<=1.
 * \param y The relative height position of the raw we want to extract. 0<=y<=1.
 */
static void update_fspt_input(layer l, network *net, float x, float y) {
    for(int input_layer_idx = 0; input_layer_idx < l.inputs;
            input_layer_idx++) {
        layer input_layer = net->layers[l.input_layers[input_layer_idx]];
        int input_w = floor(x * input_layer.out_w);
        int input_h = floor(y * input_layer.out_h);
        debug_print("input_w, input_h : (%d,%d)",
                input_w, input_h);
        debug_print("input_layer.output + input_w + l.w *input_h = %p", input_layer.output + input_w + l.w*input_h);
#ifdef GPU
        copy_gpu(input_layer.out_c, input_layer.output_gpu + input_w + l.w*input_h, input_layer.out_h*input_layer.out_w, l.fspt_input_gpu, 1);
#else
        copy_cpu(input_layer.out_c, input_layer.output + input_w + l.w*input_h, input_layer.out_h*input_layer.out_w, l.fspt_input, 1);
#endif
    }
}

/**
 * Copies the content of l.fspt_input to l.fspt_training_data[classe].
 * Make sure the content of l.fspt_input is related to the classe
 * classe.
 *
 * \param l The fspt layer.
 * \param classe The classe represented by l.fspt_input.
 */
static void copy_fspt_input_to_data(layer l, int classe) {
    size_t n = l.fspt_n_training_data[classe];
    size_t n_max = l.fspt_n_max_training_data[classe];
    if (n_max == n) realloc_fspt_data(l, classe, 0, 1);
#ifdef GPU
    //TODO: Should I use cuda_pull_array ?
    copy_gpu(l.total, l.fspt_input_gpu, 1, l.fspt_training_data[classe] + l.fspt_n_training_data[classe] * l.total, 1);
#else
    copy_cpu(l.total, l.fspt_input, 1, l.fspt_training_data[classe] + l.fspt_n_training_data[classe] * l.total, 1);
#endif
    l.fspt_n_training_data[classe] += 1;
}

/* useless : creates fspt from predicted data */
static void add_fspt_data(layer l, network net, float yolo_thresh) {
    int netw = net.w;
    int neth = net.h;
    layer yolo_layer = net.layers[l.yolo_layer];
    float *predictions = l.output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    for (int i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(int n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= yolo_thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            box bbox = get_yolo_box(predictions, yolo_layer.biases,
                    yolo_layer.mask[n], box_index, col, row,
                    l.w, l.h, netw, neth, l.w*l.h);
            for(int j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                if(prob > yolo_thresh) {
                    update_fspt_input(l, &net, bbox.x, bbox.y);
                    copy_fspt_input_to_data(l, j);
                }
            }
        }
    }
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
            box bbox = get_yolo_box(predictions, yolo_layer.biases,
                    yolo_layer.mask[n], box_index, col, row,
                    l.w, l.h, netw, neth, l.w*l.h);
            dets[count].bbox = bbox;
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                if(prob > yolo_thresh) {
                    update_fspt_input(l, net, bbox.x, bbox.y);
                    float score = fspt_get_score(l, j);
                    if(score > fspt_thresh)
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

void resize_fspt_layer(layer *l, int w, int h) {
    return;
}

void forward_fspt_layer(layer l, network net)
{
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    if(net.train_fspt) {
        for (int b = 0; b < l.batch; ++b) {
            //for(int t = 0; t < l.max_boxes; ++t){
            /* while there are truth boxes*/
            int t = 0;
            while(1) {
                box truth = float_to_box(net.truth + t*(4+1) + b*l.truths, 1);
                if(!truth.x) break;
                int class = net.truth[t*(4 + 1) + 4 + b*l.truths];
                update_fspt_input(l, &net, truth.x, truth.y);
                copy_fspt_input_to_data(l, class);
                ++t;
            }
        }
    }
}

#ifdef GPU
void forward_fspt_layer_gpu(const layer l, network net) {
    copy_gpu(l.batch*l.outputs, net.input_gpu, 1, l.output_gpu, 1);
    if(net.train_fspt) {
        for (int b = 0; b < l.batch; ++b) {
            //for(int t = 0; t < l.max_boxes; ++t){
            /* while there are truth boxes*/
            int t = 0;
            while(1) {
                box truth = float_to_box(net.truth + t*(4+1) + b*l.truths, 1);
                if(!truth.x) break;
                int class = net.truth[t*(4 + 1) + 4 + b*l.truths];
                update_fspt_input(l, &net, truth.x, truth.y);
                copy_fspt_input_to_data(l, class);
                ++t;
            }
        }
        }
        cuda_pull_array(l.output_gpu, l.output, 1);//l.batch*l.outputs);
    }
#endif

