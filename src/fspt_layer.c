#include "fspt_layer.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "fspt.h"
#include "gemm.h"
#include "utils.h"
#include "yolo_layer.h"

layer make_fspt_layer(int inputs, int *input_layers,
        int yolo_layer, network *net, float yolo_thresh,
        float *feature_limit, float *feature_importance,
        criterion_func criterion, score_func score, int batch,
        criterion_args args_template, ACTIVATION activation) {
    layer l = {0};
    l.type = FSPT;
    l.noloss = 1;
    l.onlyforward = 1;
    assert(net->layers[yolo_layer].type == YOLO);

    l.inputs = inputs;
    l.input_layers = input_layers; 
    l.classes = net->layers[yolo_layer].classes;
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
    l.truths = 90*(4 + 1);
    l.coords = 4;
    for(int i=0; i<inputs; i++) l.total += net->layers[input_layers[i]].out_c;

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.fspt_input = calloc(l.total, sizeof(float));

    l.fspts = calloc(l.classes, sizeof(fspt_t *));
    l.fspt_n_training_data = calloc(l.classes, sizeof(int));
    l.fspt_n_max_training_data = calloc(l.classes, sizeof(int));
    l.fspt_training_data = calloc(l.classes, sizeof(float *));

    l.fspt_criterion_args = args_template;

    for (int i = 0; i < l.classes; ++i) {
        l.fspts[i] = make_fspt(l.total, feature_limit, feature_importance,
                criterion, score);
    }

    l.forward = forward_fspt_layer;
#ifdef GPU
    l.forward_gpu = forward_fspt_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.fspt_input_gpu = cuda_make_array(l.fspt_input, l.total);
#endif
    l.activation = activation;

    fprintf(stderr, "fspt      %d input layer(s) : ", inputs);
    for(int i = 0; i < inputs; i++) fprintf(stderr, "%d,", input_layers[i]);
    fprintf(stderr, "    yolo layer : %d      trees : %d\n", yolo_layer,
            l.classes);

    return l;
}

/**
 * Extracts the index in l.outputs of the object in batch, at location
 * with offset entry.
 *
 * \param l The fspt layer.
 * \param batch The index of the data in the batch.
 * \param location should be l.n*l.w*l.h + row
 * \param entry index in the raw. 0 for box.x, 1 for box.y, 2 for box.w,
 *              3 for box.h, 4 for objectness, 5 + j for the prediciton
 *              of class j.
 * \return The index of the needed data. Should be used on a float pointer
 *         to the outputs of l.
 */
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
    }
    l.fspt_n_max_training_data[classe] = num;
    assert(num >= l.fspt_n_training_data[classe]);
    l.fspt_training_data[classe]
        = realloc(l.fspt_training_data[classe], l.total * num * sizeof(float));
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
 * \param b The element of the batch which is examined.
 */
static void update_fspt_input(layer l, network *net, float x, float y, int b) {
    int fspt_input_offset = 0;
    for(int input_layer_idx = 0; input_layer_idx < l.inputs;
            input_layer_idx++) {
        layer input_layer = net->layers[l.input_layers[input_layer_idx]];
        int input_w = floor(x * input_layer.out_w);
        int input_h = floor(y * input_layer.out_h);
        debug_print("input_w, input_h : (%d,%d)",
                input_w, input_h);
#ifdef GPU
        float *entry = input_layer.output_gpu
            + b * input_layer.outputs
            + input_layer.out_w*input_h
            + input_w;
        debug_print("entry = %p", entry);
        copy_gpu(input_layer.out_c, entry, input_layer.out_h*input_layer.out_w,
                l.fspt_input_gpu + fspt_input_offset, 1);
#else
        float *entry = input_layer.output
            + b * input_layer.outputs
            + input_layer.out_w*input_h
            + input_w;
        debug_print("entry = %p", entry);
        copy_cpu(input_layer.out_c, entry, input_layer.out_h*input_layer.out_w,
                l.fspt_input + fspt_input_offset, 1);
#endif
        fspt_input_offset += input_layer.out_c;
    }
#ifdef GPU
    activate_array_gpu(l.fspt_input_gpu, l.total, l.activation);
    cuda_pull_array(l.fspt_input_gpu, l.fspt_input, l.total);
#else
    activate_array(l.fspt_input, l.total, l.activation);
#endif

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
    if (n_max == n) {
        realloc_fspt_data(l, classe, 0, 1);
        debug_print("Realloc space for data (n, n_max) = (%zu, %zu)", n, n_max);
    }
    float *entry = l.fspt_training_data[classe]
        + l.fspt_n_training_data[classe] * l.total;
#ifdef GPU
    cuda_pull_array(l.fspt_input_gpu, entry, l.total);
#else
    copy_cpu(l.total, l.fspt_input, 1, entry, 1);
#endif
    if (l.total > 3)
        debug_print("add new fspt data for classe %d : %f,%f,%f,%f...",
                classe, entry[0], entry[1], entry[2], entry[3]);
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
                    update_fspt_input(l, &net, bbox.x, bbox.y, /*batch=*/0);
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
    int fspt_box_find = 0;
    int count = 0;
    for (int b = 0; b < l.batch; ++b) {
        for (i = 0; i < l.w*l.h; ++i){
            int row = i / l.w;
            int col = i % l.w;
            for(n = 0; n < l.n; ++n){
                int obj_index  = entry_index(l, b, n*l.w*l.h + i, 4);
                float objectness = predictions[obj_index];
                if(objectness <= yolo_thresh) continue;
                int box_index  = entry_index(l, b, n*l.w*l.h + i, 0);
                box bbox = get_yolo_box(predictions, yolo_layer.biases,
                        yolo_layer.mask[n], box_index, col, row,
                        l.w, l.h, netw, neth, l.w*l.h);
                for(j = 0; j < l.classes; ++j){
                    int class_index = entry_index(l, b, n*l.w*l.h + i, 4+1+j);
                    float prob = objectness*predictions[class_index];
                    if(prob > yolo_thresh) {
                        update_fspt_input(l, net, bbox.x, bbox.y, b);
                        float score = fspt_get_score(l, j);
                        if(score > fspt_thresh) {
                            fspt_box_find = 1;
                            dets[count].bbox = bbox;
                            dets[count].objectness = objectness;
                            dets[count].classes = l.classes;
                            dets[count].prob[j] = prob;
                        } else {
                            dets[count].prob[j] = 0;
                        }
                    } else {
                        dets[count].prob[j] = 0;
                    }
                }
                if (fspt_box_find) ++count;
                fspt_box_find = 0;
            }
        }
    }
    //correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

void resize_fspt_layer(layer *l, int w, int h) {
    l->w = w;
    l->h = h;
    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
#ifdef GPU
    cuda_free(l->output_gpu);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

void forward_fspt_layer(layer l, network net)
{
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    if(net.train_fspt) {
        for (int b = 0; b < l.batch; ++b) {
            //for(int t = 0; t < l.max_boxes; ++t)
            /* while there are truth boxes*/
            int t = 0;
            while(1) {
                box truth = float_to_box(net.truth + t*(4+1) + b*l.truths, 1);
                if(!truth.x) break;
                /* Get mask index */
                layer yolo = net.layers[l.yolo_layer];
                float best_iou = 0;
                int best_n = 0;
                box truth_shift = truth;
                truth_shift.x = truth_shift.y = 0;
                for(int n = 0; n < yolo.total; ++n){
                    box pred = {0};
                    pred.w = yolo.biases[2*n]/net.w;
                    pred.h = yolo.biases[2*n+1]/net.h;
                    float iou = box_iou(pred, truth_shift);
                    if (iou > best_iou){
                        best_iou = iou;
                        best_n = n;
                    }
                }
                /* Update fspt */
                int mask_n = int_index(yolo.mask, best_n, yolo.n);
                if(mask_n >= 0){
                    int class = net.truth[t*(4 + 1) + 4 + b*l.truths];
                    debug_print("truth (x,y,w,h) = (%f,%f,%f,%f) - chosen mask %d (w,h) = (%f,%f)",
                            truth.x, truth.y, truth.w, truth.h, mask_n,
                            yolo.biases[2*mask_n]/net.w,
                            yolo.biases[2*mask_n+1]/net.h);
                    update_fspt_input(l, &net, truth.x, truth.y, b);
                    copy_fspt_input_to_data(l, class);
                }
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
            //for(int t = 0; t < l.max_boxes; ++t)
            /* while there are truth boxes*/
            int t = 0;
            while(1) {
                box truth = float_to_box(net.truth + t*(4+1) + b*l.truths, 1);
                if(!truth.x) break;
                /* Get mask index */
                layer yolo = net.layers[l.yolo_layer];
                float best_iou = 0;
                int best_n = 0;
                box truth_shift = truth;
                truth_shift.x = truth_shift.y = 0;
                for(int n = 0; n < yolo.total; ++n){
                    box pred = {0};
                    pred.w = yolo.biases[2*n]/net.w;
                    pred.h = yolo.biases[2*n+1]/net.h;
                    float iou = box_iou(pred, truth_shift);
                    if (iou > best_iou){
                        best_iou = iou;
                        best_n = n;
                    }
                }
                /* Update fspt */
                int mask_n = int_index(yolo.mask, best_n, yolo.n);
                if(mask_n >= 0){
                    int class = net.truth[t*(4 + 1) + 4 + b*l.truths];
                    debug_print("truth (x,y,w,h) = (%f,%f,%f,%f) - chosen mask %d (w,h) = (%f,%f)",
                            truth.x, truth.y, truth.w, truth.h, mask_n,
                            yolo.biases[2*mask_n]/net.w,
                            yolo.biases[2*mask_n+1]/net.h);
                    update_fspt_input(l, &net, truth.x, truth.y, b);
                    copy_fspt_input_to_data(l, class);
                }
                ++t;
            }
        }
    }
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
}
#endif

void save_fspt_trees(layer l, FILE *fp) {
    for (int i = 0; i < l.classes; ++i) {
        int succ = 1;
        fspt_save_file(fp, *l.fspts[i], &succ);
    }
}

void load_fspt_trees(layer l, FILE *fp) {
    for (int i = 0; i < l.classes; ++i) {
        int succ = 1;
        fspt_load_file(fp, l.fspts[i], &succ);
    }
}

void fspt_layer_fit(layer l, int refit) {
    for (int class = 0; class < l.classes; ++class) {
        fspt_t *fspt = l.fspts[class];
        if (refit || !fspt->root) {
            if (fspt->root) free_fspt_nodes(fspt->root);
            int n = l.fspt_n_training_data[class];
            float *X = l.fspt_training_data[class];
            criterion_args *args = calloc(1, sizeof(criterion_args)); 
            *args = l.fspt_criterion_args;
            fspt_fit(n, X, args, fspt);
            free(args);
        }
#ifdef DEBUG
        if (fspt->root->type == INNER)
            print_fspt(fspt);
#endif
    }
}
