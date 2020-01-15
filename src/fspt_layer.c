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
        int yolo_layer, network *net, 
        float *feature_limit, float *feature_importance,
        criterion_func criterion, score_func score, int batch,
        criterion_args c_args_template, score_args s_args_template,
        ACTIVATION activation) {
    layer l = {0};
    l.type = FSPT;
    l.noloss = 1;
    l.onlyforward = 1;
    assert(net->layers[yolo_layer].type == YOLO);

    l.inputs = inputs;
    l.input_layers = input_layers; 
    l.classes = net->layers[yolo_layer].classes;
    l.yolo_layer = yolo_layer;

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
    l.max_boxes = net->layers[yolo_layer].max_boxes;
    l.jitter = net->layers[yolo_layer].jitter;
    for(int i=0; i<inputs; i++) l.total += net->layers[input_layers[i]].out_c;

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.fspt_input = calloc(l.total, sizeof(float));

    l.fspts = calloc(l.classes, sizeof(fspt_t *));
    l.fspt_n_training_data = calloc(l.classes, sizeof(size_t));
    l.fspt_n_max_training_data = calloc(l.classes, sizeof(size_t));
    l.fspt_training_data = calloc(l.classes, sizeof(float *));

    l.fspt_criterion_args = c_args_template;
    l.fspt_score_args = s_args_template;

    if (l.classes > 0)
        l.fspts[0] = make_fspt(l.total, feature_limit, feature_importance,
                criterion, score);
    for (int i = 1; i < l.classes; ++i) {
        float *local_feat_imp = copy_float_array(l.total, feature_importance);
        float *local_feat_lim = copy_float_array(2 * l.total, feature_limit);
        l.fspts[i] = make_fspt(l.total, local_feat_lim, local_feat_imp,
                criterion, score);
    }

    l.forward = forward_fspt_layer;
#ifdef GPU
    l.forward_gpu = forward_fspt_layer_gpu;
    if (gpu_index >= 0) {
        l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
        l.fspt_input_gpu = cuda_make_array(l.fspt_input, l.total);
    }
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
 * Realloc space for more input data corresponding to class `class` on layer l.
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
    float score = 0;
    fspt_predict(1, l.fspts[classe], l.fspt_input, &score);
    return score;
}

/**
 * Updates the row fspt_input of layer l with the content of the feature layers
 * at relative width x, height h and througth all the channels.
 * The float values pointed by l.fspt_input or l.fspt_input_gpu are modified.
 * But if GPU is defined, l.fspt_input_gpu is not pulled by this function.
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
#ifdef GPU
        float *entry = input_layer.output_gpu
            + b * input_layer.outputs
            + input_layer.out_w*input_h
            + input_w;
        copy_gpu(input_layer.out_c, entry, input_layer.out_h*input_layer.out_w,
                l.fspt_input_gpu + fspt_input_offset, 1);
#else
        float *entry = input_layer.output
            + b * input_layer.outputs
            + input_layer.out_w*input_h
            + input_w;
        copy_cpu(input_layer.out_c, entry, input_layer.out_h*input_layer.out_w,
                l.fspt_input + fspt_input_offset, 1);
#endif
        fspt_input_offset += input_layer.out_c;
    }
#ifdef GPU
    activate_array_gpu(l.fspt_input_gpu, l.total, l.activation);
    //cuda_pull_array(l.fspt_input_gpu, l.fspt_input, l.total);
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
    l.fspt_n_training_data[classe] += 1;
}


int *get_fspt_detections_batch(layer l, int w, int h, network *net,
        float yolo_thresh, float fspt_thresh, int *map, int relative,
        int suppress, detection **dets) {
    int netw = net->w;
    int neth = net->h;
    layer yolo_layer = net->layers[l.yolo_layer];
    int *count = get_yolo_detections_batch(yolo_layer, w, h, netw, neth,
            yolo_thresh, map, relative, dets);
    for (int b = 0; b < l.batch; ++b) {
        for (int i = 0; i < count[b]; ++i) {
            detection *det = dets[b] + i;
            update_fspt_input(l, net, det->bbox.x, det->bbox.y, b);
#ifdef GPU
            cuda_pull_array(l.fspt_input_gpu, l.fspt_input,
                    l.total);
#endif
            int class = max_index(det->prob, l.classes);
            det->fspt_score = fspt_get_score(l, class);
            if (suppress && det->fspt_score < fspt_thresh) {
                detection tmp_det = *det;
                dets[b][i] = dets[b][count[b] - 1];
                dets[b][count[b] - 1] = tmp_det;
                --count[b];
                --i;
            }
        }
    }
    return count;
}


void fspt_predict_truth(layer l, network net, detection **dets, int **n_boxes)
{
    int *count = calloc(l.batch, sizeof(int));
    *n_boxes = count;
    for (int b = 0; b < l.batch; ++b) {
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
                fspt_predict(1, l.fspts[class], l.fspt_input,
                        &dets[b][count[b]].fspt_score);
                dets[b][count[b]].prob[class] = 1.f;
                dets[b][count[b]].bbox.x = truth.x;
                dets[b][count[b]].bbox.y = truth.y;
                dets[b][count[b]].bbox.w = truth.w;
                dets[b][count[b]].bbox.h = truth.h;
                dets[b][count[b]].classes = l.classes;
                dets[b][count[b]].objectness = 1.f;
                ++count[b];
            }
            ++t;
        }
    }
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
        fspt_save_file(fp, *l.fspts[i], l.save_samples, &succ);
    }
}

void load_fspt_trees(layer l, FILE *fp) {
    for (int i = 0; i < l.classes; ++i) {
        int succ = 1;
        fspt_load_file(fp, l.fspts[i], l.load_samples, 1, 1, 1, &succ);
    }
}

void fspt_layer_set_samples_class(layer l, int class, int refit, int merge) {
    fspt_t *fspt = l.fspts[class];
    if (refit || !fspt->root) {
        size_t n = l.fspt_n_training_data[class];
        if (merge) {
            size_t size_base = fspt->n_samples;
            size_t max = l.fspt_n_max_training_data[class];
            if (n + size_base > max) {
                realloc_fspt_data(l, class, n + size_base, 0);
            }
            copy_cpu(size_base * l.total, fspt->samples, 1,
                    l.fspt_training_data[class] + n * l.total, 1);
            n += size_base;
        }
        float *X = l.fspt_training_data[class];
        fspt->n_samples = n;
        fspt->samples = X;
        // Commented because fit_fspt uses those values.
        //l.fspt_training_data[class] = NULL;
        //l.fspt_n_training_data[class] = 0;
        //l.fspt_n_max_training_data[class] = 0;
    }
}

void fspt_layer_rescore_class(layer l, int class) {
    fspt_t *fspt = l.fspts[class];
    assert(fspt);
    score_args *s_args = calloc(1, sizeof(score_args)); 
    *s_args = l.fspt_score_args;
    double start = what_time_is_it_now();
    fprintf(stderr, "[Fspt %s:%d]: Start rescore...\n",
            l.ref, class);
    fspt_rescore(fspt, s_args);
    long t = (what_time_is_it_now() - start) * 1000;
    fprintf(stderr,
            "[Fspt %s:%d]: rescore successful in %ldh %ldm %lds %ldms.\n",
            l.ref, class, t / (60 * 60 * 1000), t / (60 * 1000) % 60,
            t / 1000 % 60, t % 1000);
#ifdef DEBUG
    if (fspt->root->type == INNER && fspt->depth < 10)
        print_fspt(fspt);
#endif
}

void fspt_layer_fit_class(layer l, int class, int refit, int merge) {
    fspt_t *fspt = l.fspts[class];
    if (refit || !fspt->root) {
        // TODO: fix this function :
        //if (fspt->root) free_fspt_nodes(fspt->root);
        size_t n = l.fspt_n_training_data[class];
        if (merge) {
            size_t size_base = fspt->n_samples;
            size_t max = l.fspt_n_max_training_data[class];
            if (n + size_base > max) {
                realloc_fspt_data(l, class, n + size_base, 0);
            }
            copy_cpu(size_base * l.total, fspt->samples, 1,
                    l.fspt_training_data[class] + n * l.total, 1);
            n += size_base;
        }
        float *X = l.fspt_training_data[class];
        criterion_args *c_args = calloc(1, sizeof(criterion_args)); 
        score_args *s_args = calloc(1, sizeof(score_args)); 
        *c_args = l.fspt_criterion_args;
        *s_args = l.fspt_score_args;
        double start = what_time_is_it_now();
        fprintf(stderr, "[Fspt %s:%d]: Start fitting with n_samples = %ld...\n",
                l.ref, class, n);
        fspt_fit(n, X, c_args, s_args, fspt);
        l.fspt_training_data[class] = NULL;
        l.fspt_n_training_data[class] = 0;
        l.fspt_n_max_training_data[class] = 0;
        long t = (what_time_is_it_now() - start) * 1000;
        fprintf(stderr,
                "[Fspt %s:%d]: fit successful in %ldh %ldm %lds %ldms. n_nodes = %ld, depth = %d.\n",
                l.ref, class, t / (60 * 60 * 1000), t / (60 * 1000) % 60,
                t / 1000 % 60, t % 1000, fspt->n_nodes, fspt->depth);
    }
#ifdef DEBUG
    if (fspt->root->type == INNER)
        print_fspt(fspt);
#endif
}

void fspt_layer_fit(layer l, int refit, int merge) {
    for (int class = 0; class < l.classes; ++class) {
        fspt_layer_fit_class(l, class, refit, merge);
    }
    l.fspt_criterion_args = *l.fspts[0]->c_args;
    l.fspt_score_args = *l.fspts[0]->s_args;
}

void fspt_layer_rescore(layer l) {
    for (int class = 0; class < l.classes; ++class) {
        fspt_layer_rescore_class(l, class);
    }
    l.fspt_criterion_args = *l.fspts[0]->c_args;
    l.fspt_score_args = *l.fspts[0]->s_args;
}

void fspt_layer_set_samples(layer l, int refit, int merge) {
    for (int class = 0; class < l.classes; ++class) {
        fspt_layer_set_samples_class(l, class, refit, merge);
    }
}

void merge_training_data(layer l, layer base) {
    for (int class = 0; class < l.classes; ++class) {
        size_t size_l = l.fspt_n_training_data[class];
        size_t size_base = base.fspt_n_training_data[class];
        size_t max_base = base.fspt_n_max_training_data[class];
        if (size_l + size_base > max_base) {
            realloc_fspt_data(base, class, size_l, 1);
        }
        copy_cpu(size_l * l.total, l.fspt_training_data[class], 1,
                base.fspt_training_data[class] + size_base * l.total, 1);
    }
}
