#include "darknet.h"

#include <stdlib.h>

#include "box.h"
#include "image.h"
#include "utils.h"
#include "fspt_layer.h"
#include "network.h"
#include "fspt_score.h"
#include "fspt_criterion.h"

#define FLOATFORMA "% 10.5e"
#define INT_FORMAT "% 11d"

typedef struct validation_data {
    float iou_thresh;
    float fspt_thresh;
    int n_images;           // Number of images.
    int n_truth;            // Number of true boxes.
    int n_yolo_detections;  // Total number of yolo prediction.
    int classes;            // Number of class.
    /* True yolo detection */
    int tot_n_true_detection;
    int *n_true_detection;  // Size classes. n_true_detection[i] is the number
                            // of true prediction for class `i` made by yolo.
    float tot_mean_true_detection_iou;
    float *mean_true_detection_iou;
    int tot_n_true_detection_rejection;
    int *n_true_detection_rejection;
    int tot_n_true_detection_acceptance;
    int *n_true_detection_acceptance;
    float tot_mean_true_detection_rejection_fspt_score;
    float *mean_true_detection_rejection_fspt_score;
    float tot_mean_true_detection_acceptance_fspt_score;
    float *mean_true_detection_acceptance_fspt_score;
    /* Wrong class yolo detection */
    int tot_n_wrong_class_detection;
    int **n_wrong_class_detection;  // Size classes*classes.
                                    // n_wrong_class_detection[i][j] is the
                                    // number of object of true class `i`
                                    // predicted as a class `j` by yolo.
    float tot_mean_wrong_class_detection_iou;
    float **mean_wrong_class_detection_iou;
    int tot_n_wrong_class_rejection;
    int **n_wrong_class_rejection;  // Size classes*classes.
                                    // n_wrong_class_rejection[i][j] is the
                                    // number of wrong class `j` prediction
                                    // by yolo of object of true class `i` that
                                    // have a fspt score under `fspt_thresh`.
    int tot_n_wrong_class_acceptance;
    int **n_wrong_class_acceptance;  // Size classes*classes.
                                     // n_wrong_class_acceptance[i][j] is the
                                     // number of wrong class `j` prediction
                                     // by yolo of object of true class `i` that
                                     // have a fspt score above `fspt_thresh`.
    float tot_mean_wrong_class_rejection_fspt_score;
    float **mean_wrong_class_rejection_fspt_score;
    float tot_mean_wrong_class_acceptance_fspt_score;
    float **mean_wrong_class_acceptance_fspt_score;
    /* False yolo detection */
    int tot_n_false_detection;
    int *n_false_detection;  // Size classes. n_false_detection[i] is the
                             // number of prediction of class `i` by yolo while
                             // there were no object.
    int tot_n_false_detection_rejection;
    int *n_false_detection_rejection;
    int tot_n_false_detection_acceptance;
    int *n_false_detection_acceptance;
    float tot_mean_false_detection_rejection_fspt_score;
    float *mean_false_detection_rejection_fspt_score;
    float tot_mean_false_detection_acceptance_fspt_score;
    float *mean_false_detection_acceptance_fspt_score;
    /* No yolo detection */
    int tot_n_no_detection;
    int *n_no_detection;  // Size classes. n_no_detection[i] is the number of
                          // object of class `i` that were not predicted by
                          // yolo.
    /* Fspt on truth */
    int tot_n_rejection_of_truth;
    int *n_rejection_of_truth;
    int tot_n_acceptance_of_truth;
    int *n_acceptance_of_truth;
    float tot_mean_rejection_of_truth_fspt_score;
    float *mean_rejection_of_truth_fspt_score;
    float tot_mean_acceptance_of_truth_fspt_score;
    float *mean_acceptance_of_truth_fspt_score;
} validation_data;

static void print_fspt_detections(FILE **fps, char *id, detection *dets,
        int total, int classes, int w, int h) {
    for(int i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(int j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id,
                    dets[i].prob[j], xmin, ymin, xmax, ymax);
        }
    }
}

static int find_corresponding_detection(detection base, int n_dets,
        detection *comp, float iou_thresh, int *max_index,
        float *max_iou_ptr) {
    box box_base = base.bbox;
    int index = 0;
    float max_iou = 0.f;
    for (int i = 0; i < n_dets; ++i) {
        box box_comp = comp[i].bbox;
        float tmp_iou = box_iou(box_base, box_comp);
        if (tmp_iou > max_iou) {
            max_iou = tmp_iou;
            index = i;
        }
    }
    if (max_iou >= iou_thresh) {
        if (max_index) *max_index = index;
        if (max_iou_ptr) *max_iou_ptr = max_iou;
        return 1;
    } else {
        return 0;
    }
}

static void update_validation_data( int nboxes_fspt, detection *dets_fspt,
        int nboxes_truth_fspt, detection *dets_truth_fspt, int nboxes_truth,
        detection *dets_truth, validation_data *val) {
    val->n_yolo_detections += nboxes_fspt;
    val->n_truth += nboxes_truth;
    float iou_thresh = val->iou_thresh;
    float fspt_thresh = val->fspt_thresh;
    int classes = val->classes;
    int remaining_nboxes_fspt = nboxes_fspt;
    for (int i = 0; i < nboxes_truth; ++i) {
        detection det_truth = dets_truth[i];
        int class_truth = max_index(det_truth.prob, classes);
        int index = 0;
        float iou = 0.f;
        if (find_corresponding_detection(det_truth, remaining_nboxes_fspt,
                    dets_fspt, iou_thresh, &index, &iou)) {
            detection det_fspt = dets_fspt[index];
            dets_fspt[index] = dets_fspt[remaining_nboxes_fspt - 1];
            dets_fspt[remaining_nboxes_fspt - 1] = det_fspt;
            --remaining_nboxes_fspt;
            int class_yolo = max_index(det_fspt.prob, classes);
            if (class_truth == class_yolo) {
                /* True yolo detection */
                ++val->tot_n_true_detection;
                ++val->n_true_detection[class_truth];
                val->tot_mean_true_detection_iou += iou;
                val->mean_true_detection_iou[class_truth] += iou;
                if (det_fspt.fspt_score > fspt_thresh)  {
                    ++val->tot_n_true_detection_acceptance;
                    ++val->n_true_detection_acceptance[class_truth];
                    val->tot_mean_true_detection_acceptance_fspt_score
                        += det_fspt.fspt_score;
                    val->mean_true_detection_acceptance_fspt_score[class_truth]
                        += det_fspt.fspt_score;
                } else {
                    ++val->tot_n_true_detection_rejection;
                    ++val->n_true_detection_rejection[class_truth];
                    val->tot_mean_true_detection_rejection_fspt_score
                        += det_fspt.fspt_score;
                    val->mean_true_detection_rejection_fspt_score[class_truth]
                        += det_fspt.fspt_score;
                }
            } else {
                /* Wrong class yolo detection */
                ++val->tot_n_wrong_class_detection;
                ++val->n_wrong_class_detection[class_truth][class_yolo];
                val->tot_mean_wrong_class_detection_iou += iou;
                val->mean_wrong_class_detection_iou[class_truth][class_yolo]
                    += iou;
                if (det_fspt.fspt_score > fspt_thresh) {
                    ++val->tot_n_wrong_class_acceptance;
                    ++val->n_wrong_class_acceptance[class_truth][class_yolo];
                    val->tot_mean_wrong_class_acceptance_fspt_score
                        += det_fspt.fspt_score;
                    val->mean_wrong_class_acceptance_fspt_score[class_truth][class_yolo]
                        += det_fspt.fspt_score;
                } else {
                    ++val->tot_n_wrong_class_rejection;
                    ++val->n_wrong_class_rejection[class_truth][class_yolo];
                    val->tot_mean_wrong_class_rejection_fspt_score
                        += det_fspt.fspt_score;
                    val->mean_wrong_class_rejection_fspt_score[class_truth][class_yolo]
                        += det_fspt.fspt_score;
                }
            }
        } else {
            /* No yolo detection */
            ++val->n_no_detection[class_truth];
            ++val->tot_n_no_detection;
        }
        if (find_corresponding_detection(det_truth, nboxes_truth_fspt,
                    dets_truth_fspt, iou_thresh, &index, &iou)) {
            /* Fspt on truth */
            detection det_truth_fspt = dets_truth_fspt[index];
                if (det_truth_fspt.fspt_score > fspt_thresh) {
                    ++val->tot_n_acceptance_of_truth;
                    ++val->n_acceptance_of_truth[class_truth];
                    val->tot_mean_acceptance_of_truth_fspt_score
                        += det_truth_fspt.fspt_score;
                    val->mean_acceptance_of_truth_fspt_score[class_truth]
                        += det_truth_fspt.fspt_score;
                } else {
                    ++val->tot_n_rejection_of_truth;
                    ++val->n_rejection_of_truth[class_truth];
                    val->tot_mean_rejection_of_truth_fspt_score
                        += det_truth_fspt.fspt_score;
                    val->mean_rejection_of_truth_fspt_score[class_truth]
                        += det_truth_fspt.fspt_score;
                }
        }
    }
    for (int i = 0; i < remaining_nboxes_fspt; ++i) {
        /* False yolo detection */
        detection det_fspt = dets_fspt[i];
        int class_yolo = max_index(det_fspt.prob, classes);
        ++val->tot_n_false_detection;
        ++val->n_false_detection[class_yolo];
        if (det_fspt.fspt_score > fspt_thresh) {
            ++val->tot_n_false_detection_acceptance;
            ++val->n_false_detection_acceptance[class_yolo];
            val->tot_mean_false_detection_acceptance_fspt_score
                += det_fspt.fspt_score;
            val->mean_false_detection_acceptance_fspt_score[class_yolo]
                += det_fspt.fspt_score;
        } else {
            ++val->tot_n_false_detection_rejection;
            ++val->n_false_detection_rejection[class_yolo];
            val->tot_mean_false_detection_rejection_fspt_score
                += det_fspt.fspt_score;
            val->mean_false_detection_rejection_fspt_score[class_yolo]
                += det_fspt.fspt_score;
        }
    }
    /* Fspt on truth */
}

static void print_validation_data(FILE *stream, validation_data *v,
        char *title) {
    int classes = v->classes;
    /** Title **/
    if (title) {
        int len = strlen(title);
        fprintf(stream, "      ╔═");
        for (int i = 0; i < len; ++ i) fprintf(stream, "═");
        fprintf(stream, "═╗\n");
        fprintf(stream, "      ║ %s ║\n", title);
        fprintf(stream, "      ╚═");
        for (int i = 0; i < len; ++ i) fprintf(stream, "═");
        fprintf(stream, "═╝\n");
    }
    fprintf(stream, "Parameters :\n\
    -IOU threshold = %f\n\
    -FSPT threshold = %f\n\
    -Number of image = %d\n\
    -Number of true boxe = %d\n\
    -Number of yolo detection = %d\n\
    -Number of class = %d\n",
    v->iou_thresh, v->fspt_thresh, v->n_images, v->n_truth,
    v->n_yolo_detections, classes);

    /* Resume */
    fprintf(stream, "\n    ┏━━━━━━━━┓\n");
    fprintf(stream, "    ┃ RESUME ┃\n");
    fprintf(stream, "    ┗━━━━━━━━┛\n\n");
    fprintf(stream,
"             ┌────────────┬────────────┬────────────┬────────────┬────────────┐\n");
    fprintf(stream,
"             │True detect.│Wrong class │ Prediction │No detection│ FSPT score │\n");
    fprintf(stream,
"             │   by YOLO  │ prediction │while empty │   by YOLO  │  on truth  │\n");
    fprintf(stream,
"┌────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    fprintf(stream,
"│Total number│"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│\n",
        v->tot_n_true_detection, v->tot_n_wrong_class_detection,
        v->tot_n_false_detection, v->tot_n_no_detection, v->n_truth);
    fprintf(stream,
"├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    fprintf(stream,
"│  Mean IOU  │"FLOATFORMA"│"FLOATFORMA"│     //     │     //     │     //     │\n",
        v->tot_mean_true_detection_iou, v->tot_mean_wrong_class_detection_iou);
    fprintf(stream,
"├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    fprintf(stream,
"│Fspt reject │"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│     //     │"INT_FORMAT"│\n",
        v->tot_n_true_detection_rejection, v->tot_n_wrong_class_rejection,
        v->tot_n_false_detection_rejection, v->tot_n_rejection_of_truth);
    fprintf(stream,
"├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    fprintf(stream,
"│Reject score│"FLOATFORMA"│"FLOATFORMA"│"FLOATFORMA"│     //     │"FLOATFORMA"│\n",
        v->tot_mean_true_detection_rejection_fspt_score,
        v->tot_mean_wrong_class_rejection_fspt_score,
        v->tot_mean_false_detection_rejection_fspt_score,
        v->tot_mean_rejection_of_truth_fspt_score);
    fprintf(stream,
"├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    fprintf(stream,
"│Fspt accept │"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│     //     │"INT_FORMAT"│\n",
        v->tot_n_true_detection_acceptance, v->tot_n_wrong_class_acceptance,
        v->tot_n_false_detection_acceptance, v->tot_n_acceptance_of_truth);
    fprintf(stream,
"├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    fprintf(stream,
"│Accept score│"FLOATFORMA"│"FLOATFORMA"│"FLOATFORMA"│     //     │"FLOATFORMA"│\n",
        v->tot_mean_true_detection_acceptance_fspt_score,
        v->tot_mean_wrong_class_acceptance_fspt_score,
        v->tot_mean_false_detection_acceptance_fspt_score,
        v->tot_mean_acceptance_of_truth_fspt_score);
    fprintf(stream,
"└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n");
    fprintf(stream, "\n");
}

static validation_data *allocate_validation_data(int classes) {
    validation_data *v = calloc(1, sizeof(validation_data));
    /* True yolo detection */
    v->n_true_detection = calloc(classes, sizeof(int));
    v->mean_true_detection_iou = calloc(classes, sizeof(float));
    v->n_true_detection_rejection = calloc(classes, sizeof(int));
    v->n_true_detection_acceptance = calloc(classes, sizeof(int));
    v->mean_true_detection_rejection_fspt_score =
        calloc(classes, sizeof(float));
    v->mean_true_detection_acceptance_fspt_score =
        calloc(classes, sizeof(float));
    /* Wrong class yolo detection */
    v->n_wrong_class_detection = calloc(classes, sizeof(int *));
    v->mean_wrong_class_detection_iou = calloc(classes, sizeof(float *));
    v->n_wrong_class_rejection = calloc(classes, sizeof(int *));
    v->n_wrong_class_acceptance = calloc(classes, sizeof(int *));
    v->mean_wrong_class_rejection_fspt_score =
        calloc(classes, sizeof(float *));
    v->mean_wrong_class_acceptance_fspt_score =
        calloc(classes, sizeof(float *));
    for (int i = 0; i < classes; ++i) {
        v->n_wrong_class_detection[i] = calloc(classes, sizeof(int));
        v->mean_wrong_class_detection_iou[i] = calloc(classes, sizeof(float));
        v->n_wrong_class_rejection[i] = calloc(classes, sizeof(int));
        v->n_wrong_class_acceptance[i] = calloc(classes, sizeof(int));
        v->mean_wrong_class_rejection_fspt_score[i] =
            calloc(classes, sizeof(float));
        v->mean_wrong_class_acceptance_fspt_score[i] =
            calloc(classes, sizeof(float));
    }
    /* False yolo detection */
    v->n_false_detection = calloc(classes, sizeof(int));
    v->n_false_detection_rejection = calloc(classes, sizeof(int));
    v->n_false_detection_acceptance = calloc(classes, sizeof(int));
    v->mean_false_detection_rejection_fspt_score =
        calloc(classes, sizeof(float));
    v->mean_false_detection_acceptance_fspt_score =
        calloc(classes, sizeof(float));
    /* No yolo detection */
    v->n_no_detection = calloc(classes, sizeof(int));
    /* Fspt on truth */
    v->n_rejection_of_truth = calloc(classes, sizeof(int));
    v->n_acceptance_of_truth = calloc(classes, sizeof(int));
    v->mean_rejection_of_truth_fspt_score = calloc(classes, sizeof(float));
    v->mean_acceptance_of_truth_fspt_score = calloc(classes, sizeof(float));

    return v;
}

static void free_validation_data(validation_data *v) {
    /* True yolo detection */
    free(v->n_true_detection);
    free(v->mean_true_detection_iou);
    free(v->n_true_detection_rejection);
    free(v->n_true_detection_acceptance);
    free(v->mean_true_detection_rejection_fspt_score);
    free(v->mean_true_detection_acceptance_fspt_score);
    /* Wrong class yolo detection */
    for (int i = 0; i < v->classes; ++i) {
        free(v->n_wrong_class_detection[i]);
        free(v->mean_wrong_class_detection_iou[i]);
        free(v->n_wrong_class_rejection[i]);
        free(v->n_wrong_class_acceptance[i]);
        free(v->mean_wrong_class_rejection_fspt_score[i]);
        free(v->mean_wrong_class_acceptance_fspt_score[i]);
    }
    free(v->n_wrong_class_detection);
    free(v->mean_wrong_class_detection_iou);
    free(v->n_wrong_class_rejection);
    free(v->n_wrong_class_acceptance);
    free(v->mean_wrong_class_rejection_fspt_score);
    free(v->mean_wrong_class_acceptance_fspt_score);
    /* False yolo detection */
    free(v->n_false_detection);
    free(v->n_false_detection_rejection);
    free(v->n_false_detection_acceptance);
    free(v->mean_false_detection_rejection_fspt_score);
    free(v->mean_false_detection_acceptance_fspt_score);
    /* No yolo detection */
    free(v->n_no_detection);
    /* Fspt on truth */
    free(v->n_rejection_of_truth);
    free(v->n_acceptance_of_truth);
    free(v->mean_rejection_of_truth_fspt_score);
    free(v->mean_acceptance_of_truth_fspt_score);
    /* Free validation data */
    free(v);
}

static void print_stats(char *datacfg, char *cfgfile, char *weightfile,
        float yolo_thresh, float fspt_thresh) {
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);

    list *fspt_layers = get_network_layers_by_type(net, FSPT);
    while (fspt_layers->size > 0) {
        layer *l = (layer *) list_pop(fspt_layers);
        for (int i = 0; i < l->classes; ++i) {
            fspt_t *fspt = l->fspts[i];
            fspt_stats *stats = get_fspt_stats(fspt, 0, NULL);
            char buf[256] = {0};
            sprintf(buf, "%s class %s", l->ref, names[i]);
            print_fspt_criterion_args(stderr, &l->fspt_criterion_args, buf);
            print_fspt_score_args(stderr, &l->fspt_score_args, NULL);
            print_fspt_stats(stderr, stats, NULL);
            free_fspt_stats(stats);
        }
    }
}

void test_fspt(char *datacfg, char *cfgfile, char *weightfile, char *filename,
        float yolo_thresh, float fspt_thresh, float hier_thresh, char *outfile,
        int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input,
                what_time_is_it_now()-time);
        int nboxes_yolo = 0;
        detection *dets_yolo = get_network_boxes(net, im.w, im.h, yolo_thresh,
                hier_thresh, 0, 1, &nboxes_yolo);
        printf("%d boxes predicted by yolo.\n", nboxes_yolo);
        if (nms) do_nms_sort(dets_yolo, nboxes_yolo, l.classes, nms);
        draw_detections(im, dets_yolo, nboxes_yolo, yolo_thresh, names,
                alphabet, l.classes);
        int nboxes_fspt = 0;
        detection *dets_fspt = get_network_fspt_boxes(net, im.w, im.h,
                yolo_thresh, fspt_thresh, hier_thresh, 0, 1, 0, &nboxes_fspt);
        printf("%d boxes predicted by fspt.\n", nboxes_fspt);
        if (nms) do_nms_sort(dets_fspt, nboxes_fspt, l.classes, nms);
        draw_fspt_detections(im, dets_fspt, nboxes_fspt, yolo_thresh, names,
                alphabet, l.classes);
        free_detections(dets_yolo, nboxes_yolo);
        free_detections(dets_fspt, nboxes_fspt);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions_fspt");
#ifdef OPENCV
            make_window("predictions_fspt", 512, 512, 0);
            show_image(im, "predictions_fspt", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

static void train_fspt(char *datacfg, char *cfgfile, char *weightfile,
        int *gpus, int ngpus, int clear, int refit, int ordered,
        int one_thread, int merge, int only_fit, int print_stats_after_fit) {
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.txt");
    char *backup_directory = option_find_str(options, "backup", "backup/");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    data train, buffer;

    layer l = net->layers[0];
    for (int i = 0; i < net->n; ++i) {
        l = net->layers[i];
        if (l.type == FSPT || l.type == YOLO)
            break;
    }
    if (l.type != FSPT && l.type != YOLO)
        error("The net must have fspt or yolo layers");

    int classes = l.classes;

    if (only_fit) { 
        fprintf(stderr, "Only fit FSPTs...\n");
    } else {
        list *plist = get_paths(train_images);
        char **paths = (char **)list_to_array(plist);

        load_args args = get_base_args(net);
        args.coords = l.coords;
        args.paths = paths;
        args.n = imgs;
        args.m = plist->size;
        args.classes = classes;
        args.jitter = l.jitter;
        args.num_boxes = l.max_boxes;
        args.d = &buffer;
        args.type = DETECTION_DATA;
        args.threads = 64;
        args.ordered = ordered;
        args.beg = 0;

        pthread_t load_thread = load_data(args);
        double time;
        if (ordered) {
            net->max_batches = plist->size / imgs;
        }
        while (get_current_batch(net) < net->max_batches) {
            time=what_time_is_it_now();
            pthread_join(load_thread, 0);
            train = buffer;
            args.beg = *net->seen;
            load_thread = load_data(args);
            printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
            time=what_time_is_it_now();
#ifdef GPU
            if(ngpus == 1){
                train_network_fspt(net, train);
            } else {
                train_networks_fspt(nets, ngpus, train, 4);
            }
#else
            train_network_fspt(net, train);
#endif
            i = get_current_batch(net);
            fprintf(stderr,
                    "%ld: %lf seconds, %d images added to fspt input\n",
                    get_current_batch(net), what_time_is_it_now()-time,
                    i*imgs);
            free_data(train);
        }
#ifdef GPU
        if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
        fspt_layers_set_samples(net, refit, merge);
        char buff[256];
        sprintf(buff, "%s/%s_data_extraction.weights", backup_directory, base);
        save_weights(net, buff);
        fprintf(stderr, "Data extraction done. Fitting FSPTs...\n");
    }
    fit_fspts(net, classes, refit, one_thread, merge);
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    fprintf(stderr, "End of FSPT training\n");
    if (print_stats_after_fit) {
        list *fspt_layers = get_network_layers_by_type(net, FSPT);
        while (fspt_layers->size > 0) {
            layer *l = (layer *) list_pop(fspt_layers);
            for (int i = 0; i < l->classes; ++i) {
                fspt_t *fspt = l->fspts[i];
                fspt_stats *stats = get_fspt_stats(fspt, 0, NULL);
                char buf[256] = {0};
                sprintf(buf, "%s class %s", l->ref, names[i]);
                print_fspt_criterion_args(stderr, &l->fspt_criterion_args, buf);
                print_fspt_score_args(stderr, &l->fspt_score_args, NULL);
                print_fspt_stats(stderr, stats, NULL);
                free_fspt_stats(stats);
            }
        }
    }
}

static void validate_fspt(char *datacfg, char *cfgfile, char *weightfile,
        float yolo_thresh, float fspt_thresh, float hier_thresh, int ngpus,
        int ordered, char *outfile) {
    //TODO
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/valid.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *base = basecfg(cfgfile);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network **nets = calloc(ngpus, sizeof(network));
    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, 1);
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    data val, buffer;

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[0];
    for (int i = 0; i < net->n; ++i) {
        l = net->layers[i];
        if (l.type == FSPT || l.type == YOLO)
            break;
    }
    if (l.type != FSPT && l.type != YOLO)
        error("The net must have fspt or yolo layers");

    int classes = l.classes;

    double start = what_time_is_it_now();

    float nms = .45;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = 0;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 64;
    args.ordered = 1;
    args.beg = 0;

    validation_data *val_data = allocate_validation_data(classes);
    val_data->n_images = plist->size;

    pthread_t load_thread = load_data(args);
    double time;
    if (ordered) {
        net->max_batches = plist->size / imgs;
    }
    while (get_current_batch(net) < net->max_batches) {
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        val = buffer;
        args.beg = *net->seen;
        load_thread = load_data(args);
        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time=what_time_is_it_now();
#ifdef GPU
        if(ngpus == 1){
            validate_network_fspt(net, val);
        } else {
            validate_network_fspt(net, val);
            validate_networks_fspt(nets, ngpus, val, 4);
        }
#else
        validate_network_fspt(net, val);
#endif
        i = get_current_batch(net);
        fprintf(stderr,
                "%ld: %lf seconds, %d images added to validation.\n",
                get_current_batch(net), what_time_is_it_now()-time,
                i*imgs);
        int w = val.w; // ARE YOU SURE ?
        int h = val.h;
        /* FSPT boxes. */
        int *nboxes_fspt;
        detection **dets_fspt = get_network_fspt_boxes_batch(net, w, h,
                yolo_thresh, fspt_thresh, hier_thresh, map, 0, 0,
                &nboxes_fspt);
        if (nms) {
            for (int b = 0; b < net->batch; ++b)
                do_nms_sort(dets_fspt[b], nboxes_fspt[b], classes, nms);
        }
        /* FSPT truth boxes */
        int *nboxes_truth_fspt;
        detection **dets_truth_fspt =
            get_network_fspt_truth_boxes_batch(net, w, h,
                yolo_thresh, fspt_thresh, hier_thresh, map, 0,
                &nboxes_truth_fspt);
        /* Truth boxes */
        int *nboxes_truth;
        detection **dets_truth = get_network_truth_boxes_batch(net, w, h,
                &nboxes_truth);

        for (int b = 0; b < net->batch; ++b) {
            update_validation_data(nboxes_fspt[b], dets_fspt[b],
                    nboxes_truth_fspt[b], dets_truth_fspt[b],
                    nboxes_truth[b], dets_truth[b], val_data);
            free_detections(dets_fspt[b], nboxes_fspt[b]);
            free_detections(dets_truth_fspt[b], nboxes_truth_fspt[b]);
            free_detections(dets_truth[b], nboxes_truth[b]);
            free(dets_fspt);
            free(dets_truth_fspt);
            free(dets_truth);
            free(nboxes_fspt);
            free(nboxes_truth_fspt);
            free(nboxes_truth);
        }
        free_data(val);
    }
    print_validation_data(stderr, val_data, "VALIDATION RESULT");
    free_validation_data(val_data);
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    fprintf(stderr, "Total Detection Time: %f Seconds\n",
            what_time_is_it_now() - start);
}

static void validate_fspt_recall(char *cfgfile, char *weightfile) {
    //TODO
    error("TODO");
}

void run_fspt(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr,
"usage: %s %s <train/test/valid> <datacfg> <netcfg> [weights] [inputfile] [options]\n\
With :\n\
    train -> train fspts on an already trained yolo network.\n\
    test  -> test fspt predictions.\n\
    valid -> validate fspt.\n\
And :\n\
    <datacfg>   -> path to the data configuration file.\n\
    <netcfg>    -> path to the network configuration file.\n\
    [weights]   -> path to a weightfile corresponding to the netcfg file.\n\
                   (optional)\n\
    [inputfile] -> path to a file with list of test images.\n\
                   only for commande test. (optional)\n\
Options are :\n\
    -yolo_thresh -> yolo detection threashold. default 0.5.\n\
    -fspt_thresh -> fspt rejection threshold. default 0.5.\n\
    -hier        -> unused.\n\
    -gpus        -> coma separated list of gpus.\n\
    -out         -> ouput file for test and valid.\n\
    -clear       -> if set, the training number of seen images is reset.\n\
    -refit       -> if set, the fspts are refitted if they already exist.\n\
    -ordered     -> if set, the data are selected sequentialy and not randomly.\n\
    -merge       -> if set, newly extracted data are merged to existing.\n\
    -only_fit    -> if set, don't extract new data. implies -merge.\n\
    -one_thread  -> if set, the fspts are fitted in only one thread instead of\n\
                    one thread per fspt.\n\
    -fullscreen  -> unused.\n\
    -print_stats -> if set, print the statistics of the fspts after training.\n",
                argv[0], argv[1]);
        return;
    }

    float yolo_thresh = find_float_arg(argc, argv, "-yolo_thresh", .5);
    float fspt_thresh = find_float_arg(argc, argv, "-fspt_thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int clear = find_arg(argc, argv, "-clear");
    int refit_fspts = find_arg(argc, argv, "-refit");
    int ordered = find_arg(argc, argv, "-ordered");
    int one_thread = find_arg(argc, argv, "-one_thread");
    int only_fit = find_arg(argc, argv, "-only_fit");
    int merge = find_arg(argc, argv, "-merge") || only_fit;
    int print_stats_after_fit = find_arg(argc, argv, "-print_stats");
    int fullscreen = find_arg(argc, argv, "-fullscreen");

    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test"))
        test_fspt(datacfg, cfg, weights, filename, yolo_thresh, fspt_thresh,
                hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train"))
        train_fspt(datacfg, cfg, weights, gpus, ngpus, clear, refit_fspts,
                ordered, one_thread, merge, only_fit, print_stats_after_fit);
    else if(0==strcmp(argv[2], "valid"))
        validate_fspt(datacfg, cfg, weights, yolo_thresh, fspt_thresh,
                hier_thresh, ngpus, ordered, outfile);
    else if(0==strcmp(argv[2], "recall"))
        validate_fspt_recall(cfg, weights);
    else if (0 == strcmp(argv[2], "stats"))
        print_stats(datacfg, cfg, weights, yolo_thresh, fspt_thresh);
}

#undef FLOATFORMA
#undef INT_FORMAT
