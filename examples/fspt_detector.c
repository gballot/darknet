#include "darknet.h"

#include <assert.h>
#include <stdlib.h>

#include "box.h"
#include "image.h"
#include "utils.h"
#include "fspt_layer.h"
#include "network.h"
#include "fspt_score.h"
#include "fspt_criterion.h"

#define FLT_FORMAT "%12g"
#define INT_FORMAT "%12d"
#define N_CORES 24

typedef struct validation_data {
    float iou_thresh;       // Intersection Over Union threshold.
    float fspt_thresh;      // FSPT threshold for this validation.
    int n_images;           // Number of images.
    int n_yolo_detections;  // Total number of yolo prediction.
    int classes;            // Number of classes.
    char **names;           // Size classes. Name of the classes.
    /* True yolo detection */
    int tot_n_true_detection; // Number of true prediction by Yolo.
    int *n_true_detection;  // Size classes. n_true_detection[i] is the number
                            // of true prediction for class `i` made by yolo.
    float tot_sum_true_detection_iou;  // Divided by tot_n_true_detection,
                                       // gives the average IOU of true
                                       // detections.
    float *sum_true_detection_iou;  // i^th element divided by
                                    //  n_ture_detection[i] gives the average
                                    //  IOU for the true detections of class i.
    int tot_n_true_detection_rejection;
    int *n_true_detection_rejection;
    int tot_n_true_detection_acceptance;
    int *n_true_detection_acceptance;
    float tot_sum_true_detection_rejection_fspt_score;
    float *sum_true_detection_rejection_fspt_score;
    float tot_sum_true_detection_acceptance_fspt_score;
    float *sum_true_detection_acceptance_fspt_score;
    /* Wrong class yolo detection */
    int tot_n_wrong_class_detection;
    int **n_wrong_class_detection;  // Size classes*classes.
                                    // n_wrong_class_detection[i][j] is the
                                    // number of object of true class `i`
                                    // predicted as a class `j` by yolo.
    float tot_sum_wrong_class_detection_iou;
    float **sum_wrong_class_detection_iou;
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
    float tot_sum_wrong_class_rejection_fspt_score;
    float **sum_wrong_class_rejection_fspt_score;
    float tot_sum_wrong_class_acceptance_fspt_score;
    float **sum_wrong_class_acceptance_fspt_score;
    /* False yolo detection */
    int tot_n_false_detection;
    int *n_false_detection;  // Size classes. n_false_detection[i] is the
                             // number of prediction of class `i` by yolo while
                             // there were no object.
    int tot_n_false_detection_rejection;
    int *n_false_detection_rejection;
    int tot_n_false_detection_acceptance;
    int *n_false_detection_acceptance;
    float tot_sum_false_detection_rejection_fspt_score;
    float *sum_false_detection_rejection_fspt_score;
    float tot_sum_false_detection_acceptance_fspt_score;
    float *sum_false_detection_acceptance_fspt_score;
    /* No yolo detection */
    int tot_n_no_detection;
    int *n_no_detection;  // Size classes. n_no_detection[i] is the number of
                          // object of class `i` that were not predicted by
                          // yolo.
    float tot_sum_no_detection_iou;
    float *sum_no_detection_iou;
    /* Fspt on truth */
    int tot_n_truth;        // Number of true boxes.
    int *n_truth;           // Size classes. Number of true boxes per class.
    int tot_n_rejection_of_truth;
    int *n_rejection_of_truth;
    int tot_n_acceptance_of_truth;
    int *n_acceptance_of_truth;
    float tot_sum_rejection_of_truth_fspt_score;
    float *sum_rejection_of_truth_fspt_score;
    float tot_sum_acceptance_of_truth_fspt_score;
    float *sum_acceptance_of_truth_fspt_score;
} validation_data;

static int find_corresponding_detection(detection base, int n_dets,
        detection *comp, float iou_thresh, int *max_index,
        float *max_iou_ptr) {
    if (!n_dets) return 0;
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
        debug_print("find corresponding detection index = %d, iou = %f", index,
                max_iou);
        return 1;
    } else {
        debug_print("don't find corresponding detection iou = %f", max_iou);
        return 0;
    }
}

static void update_validation_data( int nboxes_fspt, detection *dets_fspt,
        int nboxes_truth, detection *dets_truth,
        validation_data *val) {
    debug_print("nboxes_fspt = %d, nboxes_truth = %d",
            nboxes_fspt, nboxes_truth);
    val->n_yolo_detections += nboxes_fspt;
    val->tot_n_truth += nboxes_truth;
    float iou_thresh = val->iou_thresh;
    float fspt_thresh = val->fspt_thresh;
    int classes = val->classes;
    int remaining_nboxes_fspt = nboxes_fspt;
    for (int i = 0; i < nboxes_truth; ++i) {
        detection det_truth = dets_truth[i];
        int class_truth = max_index(det_truth.prob, classes);
        ++val->n_truth[class_truth];
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
                val->tot_sum_true_detection_iou += iou;
                val->sum_true_detection_iou[class_truth] += iou;
                if (det_fspt.fspt_score > fspt_thresh)  {
                    ++val->tot_n_true_detection_acceptance;
                    ++val->n_true_detection_acceptance[class_truth];
                    val->tot_sum_true_detection_acceptance_fspt_score
                        += det_fspt.fspt_score;
                    val->sum_true_detection_acceptance_fspt_score[class_truth]
                        += det_fspt.fspt_score;
                } else {
                    ++val->tot_n_true_detection_rejection;
                    ++val->n_true_detection_rejection[class_truth];
                    val->tot_sum_true_detection_rejection_fspt_score
                        += det_fspt.fspt_score;
                    val->sum_true_detection_rejection_fspt_score[class_truth]
                        += det_fspt.fspt_score;
                }
            } else {
                /* Wrong class yolo detection */
                ++val->tot_n_wrong_class_detection;
                ++val->n_wrong_class_detection[class_truth][class_yolo];
                val->tot_sum_wrong_class_detection_iou += iou;
                val->sum_wrong_class_detection_iou[class_truth][class_yolo]
                    += iou;
                if (det_fspt.fspt_score > fspt_thresh) {
                    ++val->tot_n_wrong_class_acceptance;
                    ++val->n_wrong_class_acceptance[class_truth][class_yolo];
                    val->tot_sum_wrong_class_acceptance_fspt_score
                        += det_fspt.fspt_score;
                    val->sum_wrong_class_acceptance_fspt_score[class_truth][class_yolo]
                        += det_fspt.fspt_score;
                } else {
                    ++val->tot_n_wrong_class_rejection;
                    ++val->n_wrong_class_rejection[class_truth][class_yolo];
                    val->tot_sum_wrong_class_rejection_fspt_score
                        += det_fspt.fspt_score;
                    val->sum_wrong_class_rejection_fspt_score[class_truth][class_yolo]
                        += det_fspt.fspt_score;
                }
            }
        } else {
            /* No yolo detection */
            ++val->tot_n_no_detection;
            ++val->n_no_detection[class_truth];
        }
        /* Fspt on truth */
        if (det_truth.fspt_score > fspt_thresh) {
            ++val->tot_n_acceptance_of_truth;
            ++val->n_acceptance_of_truth[class_truth];
            val->tot_sum_acceptance_of_truth_fspt_score
                += det_truth.fspt_score;
            val->sum_acceptance_of_truth_fspt_score[class_truth]
                += det_truth.fspt_score;
        } else {
            ++val->tot_n_rejection_of_truth;
            ++val->n_rejection_of_truth[class_truth];
            val->tot_sum_rejection_of_truth_fspt_score
                += det_truth.fspt_score;
            val->sum_rejection_of_truth_fspt_score[class_truth]
                += det_truth.fspt_score;
        }
    }
    for (int i = 0; i < remaining_nboxes_fspt; ++i) {
        /* False yolo detection */
        detection det_fspt = dets_fspt[i];
        int class_yolo = max_index(det_fspt.prob, classes);
        int index = 0;
        float iou = 0.f;
        if (find_corresponding_detection(det_fspt, nboxes_truth,
                    dets_truth, iou_thresh, &index, &iou)) {
            val->tot_sum_no_detection_iou += iou;
            val->sum_no_detection_iou[class_yolo] += iou;
        }
        ++val->tot_n_false_detection;
        ++val->n_false_detection[class_yolo];
        if (det_fspt.fspt_score > fspt_thresh) {
            ++val->tot_n_false_detection_acceptance;
            ++val->n_false_detection_acceptance[class_yolo];
            val->tot_sum_false_detection_acceptance_fspt_score
                += det_fspt.fspt_score;
            val->sum_false_detection_acceptance_fspt_score[class_yolo]
                += det_fspt.fspt_score;
        } else {
            ++val->tot_n_false_detection_rejection;
            ++val->n_false_detection_rejection[class_yolo];
            val->tot_sum_false_detection_rejection_fspt_score
                += det_fspt.fspt_score;
            val->sum_false_detection_rejection_fspt_score[class_yolo]
                += det_fspt.fspt_score;
        }
    }
}

static void print_validation_data(FILE *stream, validation_data *v,
        int only_resume, char *title) {
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
    if (!v) {
        fprintf(stream, "No validation data.\n\n");
        return;
    }

    int classes = v->classes;
    if (!only_resume) {
        fprintf(stream, "\
Parameters :\n\
    -IOU threshold = %f\n\
    -FSPT threshold = %f\n\
    -Number of image = %d\n\
    -Number of true boxe = %d\n\
    -Number of yolo detection = %d\n\
    -Number of class = %d\n\n",
        v->iou_thresh, v->fspt_thresh, v->n_images, v->tot_n_truth,
        v->n_yolo_detections, classes);
    }

    /* Resume */
    if (!only_resume) {
        fprintf(stream, "    ┏━━━━━━━━┓\n");
        fprintf(stream, "    ┃ RESUME ┃\n");
        fprintf(stream, "    ┗━━━━━━━━┛\n\n");
    }
    fprintf(stream, "\
                      ┌────────────┬────────────┬────────────┬────────────┬────────────┐\n\
                      │True detect.│Wrong class │ Prediction │No detection│    True    │\n\
                      │   by YOLO  │ prediction │while empty │   by YOLO  │    boxes   │\n\
┌─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│   Total numberer    │"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│      Mean IOU       │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│     //     │     //     │\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│   Fspt rejection    │"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│     //     │"INT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│   Rejection score   │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│     //     │"FLT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│   Fspt acceptance   │"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│     //     │"INT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│  Acceptance score   │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│     //     │"FLT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│Rejection proportion │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│     //     │"FLT_FORMAT"│\n\
└─────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n\n",
        // Total number
        v->tot_n_true_detection, v->tot_n_wrong_class_detection,
        v->tot_n_false_detection, v->tot_n_no_detection, v->tot_n_truth,
        // Mean iou
        safe_divd(v->tot_sum_true_detection_iou, v->tot_n_true_detection),
        safe_divd(v->tot_sum_wrong_class_detection_iou,
            v->tot_n_wrong_class_detection),
        safe_divd(v->tot_sum_no_detection_iou, v->tot_n_false_detection),
        // Fspt rejection
        v->tot_n_true_detection_rejection, v->tot_n_wrong_class_rejection,
        v->tot_n_false_detection_rejection, v->tot_n_rejection_of_truth,
        // Rejection score
        safe_divd(v->tot_sum_true_detection_rejection_fspt_score,
            v->tot_n_true_detection_rejection),
        safe_divd(v->tot_sum_wrong_class_rejection_fspt_score,
            v->tot_n_wrong_class_rejection),
        safe_divd(v->tot_sum_false_detection_rejection_fspt_score,
            v->tot_n_false_detection_rejection),
        safe_divd(v->tot_sum_rejection_of_truth_fspt_score,
            v->tot_n_rejection_of_truth),
        // Fspt acceptance
        v->tot_n_true_detection_acceptance, v->tot_n_wrong_class_acceptance,
        v->tot_n_false_detection_acceptance, v->tot_n_acceptance_of_truth,
        // Acceptance score
        safe_divd(v->tot_sum_true_detection_acceptance_fspt_score,
            v->tot_n_true_detection_acceptance),
        safe_divd(v->tot_sum_wrong_class_acceptance_fspt_score,
            v->tot_n_wrong_class_acceptance),
        safe_divd(v->tot_sum_false_detection_acceptance_fspt_score,
                v->tot_n_false_detection_acceptance),
        safe_divd(v->tot_sum_acceptance_of_truth_fspt_score,
                v->tot_n_acceptance_of_truth),
        // Rejection proportion
        safe_divd(v->tot_n_true_detection_rejection, v->tot_n_true_detection),
        safe_divd(v->tot_n_wrong_class_rejection,
                v->tot_n_wrong_class_detection),
        safe_divd(v->tot_n_false_detection_rejection,
                v->tot_n_false_detection),
        safe_divd(v->tot_n_rejection_of_truth, v->tot_n_truth)
        );

    if (only_resume) return;

    /* Per class */
    fprintf(stream, "    ┏━━━━━━━━━━━┓\n");
    fprintf(stream, "    ┃ PER CLASS ┃\n");
    fprintf(stream, "    ┗━━━━━━━━━━━┛\n\n");
    for (int i = 0; i < classes; ++i) {
        fprintf(stream, "\
                      ┌────────────────────────────────────────────────────────────────┐\n\
                      │ Class : %54s │\n\
                      ├────────────┬────────────┬────────────┬────────────┬────────────┤\n\
                      │True detect.│Wrong class │ Prediction │No detection│    True    │\n\
                      │   by YOLO  │ prediction │while empty │   by YOLO  │    boxes   │\n\
┌─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│   Total numberer    │"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│      Mean IOU       │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│     //     │     //     │\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│   Fspt rejection    │"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│     //     │"INT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│   Rejection score   │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│     //     │"FLT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│   Fspt acceptance   │"INT_FORMAT"│"INT_FORMAT"│"INT_FORMAT"│     //     │"INT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│  Acceptance score   │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│     //     │"FLT_FORMAT"│\n\
├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│Rejection proportion │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│     //     │"FLT_FORMAT"│\n\
└─────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n\n",
                // Class name
                v->names[i],
                // Total number
                v->n_true_detection[i],
                sum_array_int(v->n_wrong_class_detection[i], classes),
                v->n_false_detection[i], v->n_no_detection[i], v->n_truth[i],
                // Mean iou
                safe_divd(v->sum_true_detection_iou[i],
                        v->n_true_detection[i]),
                safe_divd(
                        sum_array(v->sum_wrong_class_detection_iou[i],
                            classes),
                        sum_array_int(v->n_wrong_class_detection[i], classes)
                        ),
                safe_divd(v->sum_no_detection_iou[i],
                        v->n_false_detection[i]),
                // Fspt rejection
                v->n_true_detection_rejection[i],
                sum_array_int(v->n_wrong_class_rejection[i], classes),
                v->n_false_detection_rejection[i],
                v->n_rejection_of_truth[i],
                // Rejection score
                safe_divd(v->sum_true_detection_rejection_fspt_score[i],
                        v->n_true_detection_rejection[i]),
                safe_divd(
                        sum_array(v->sum_wrong_class_rejection_fspt_score[i],
                            classes),
                        sum_array_int(v->n_wrong_class_rejection[i], classes)
                        ),
                safe_divd(v->sum_false_detection_rejection_fspt_score[i],
                        v->n_false_detection_rejection[i]),
                safe_divd(v->sum_rejection_of_truth_fspt_score[i],
                        v->n_rejection_of_truth[i]),
                // Fspt acceptance
                v->n_true_detection_acceptance[i],
                sum_array_int(v->n_wrong_class_acceptance[i], classes),
                v->n_false_detection_acceptance[i],
                v->n_acceptance_of_truth[i],
                // Acceptance score
                safe_divd(v->sum_true_detection_acceptance_fspt_score[i],
                        v->n_true_detection_acceptance[i]),
                safe_divd(
                        sum_array(v->sum_wrong_class_acceptance_fspt_score[i],
                            classes),
                        sum_array_int(v->n_wrong_class_acceptance[i], classes)
                        ),
                safe_divd(v->sum_false_detection_acceptance_fspt_score[i],
                        v->n_false_detection_acceptance[i]),
                safe_divd(v->sum_acceptance_of_truth_fspt_score[i],
                        v->n_acceptance_of_truth[i]),
                // Rejection proportion
                safe_divd(v->n_true_detection_rejection[i],
                        v->n_true_detection[i]),
                safe_divd(
                        sum_array_int(v->n_wrong_class_rejection[i], classes),
                        sum_array_int(v->n_wrong_class_detection[i], classes)
                        ),
                safe_divd(v->n_false_detection_rejection[i],
                        v->n_false_detection[i]),
                safe_divd(v->n_rejection_of_truth[i], v->n_truth[i])
                    );
    }
}

static validation_data *allocate_validation_data(int classes, char **names) {
    validation_data *v = calloc(1, sizeof(validation_data));
    v->classes = classes;
    v->names = calloc(classes, sizeof(char *));
    for (int i = 0; i < classes; ++i) {
        v->names[i] = copy_string(names[i]);
    }
    /* True yolo detection */
    v->n_true_detection = calloc(classes, sizeof(int));
    v->sum_true_detection_iou = calloc(classes, sizeof(float));
    v->n_true_detection_rejection = calloc(classes, sizeof(int));
    v->n_true_detection_acceptance = calloc(classes, sizeof(int));
    v->sum_true_detection_rejection_fspt_score =
        calloc(classes, sizeof(float));
    v->sum_true_detection_acceptance_fspt_score =
        calloc(classes, sizeof(float));
    /* Wrong class yolo detection */
    v->n_wrong_class_detection = calloc(classes, sizeof(int *));
    v->sum_wrong_class_detection_iou = calloc(classes, sizeof(float *));
    v->n_wrong_class_rejection = calloc(classes, sizeof(int *));
    v->n_wrong_class_acceptance = calloc(classes, sizeof(int *));
    v->sum_wrong_class_rejection_fspt_score =
        calloc(classes, sizeof(float *));
    v->sum_wrong_class_acceptance_fspt_score =
        calloc(classes, sizeof(float *));
    for (int i = 0; i < classes; ++i) {
        v->n_wrong_class_detection[i] = calloc(classes, sizeof(int));
        v->sum_wrong_class_detection_iou[i] = calloc(classes, sizeof(float));
        v->n_wrong_class_rejection[i] = calloc(classes, sizeof(int));
        v->n_wrong_class_acceptance[i] = calloc(classes, sizeof(int));
        v->sum_wrong_class_rejection_fspt_score[i] =
            calloc(classes, sizeof(float));
        v->sum_wrong_class_acceptance_fspt_score[i] =
            calloc(classes, sizeof(float));
    }
    /* False yolo detection */
    v->n_false_detection = calloc(classes, sizeof(int));
    v->n_false_detection_rejection = calloc(classes, sizeof(int));
    v->n_false_detection_acceptance = calloc(classes, sizeof(int));
    v->sum_false_detection_rejection_fspt_score =
        calloc(classes, sizeof(float));
    v->sum_false_detection_acceptance_fspt_score =
        calloc(classes, sizeof(float));
    /* No yolo detection */
    v->n_no_detection = calloc(classes, sizeof(int));
    v->sum_no_detection_iou = calloc(classes, sizeof(float));
    /* Fspt on truth */
    v->n_truth = calloc(classes, sizeof(int));
    v->n_rejection_of_truth = calloc(classes, sizeof(int));
    v->n_acceptance_of_truth = calloc(classes, sizeof(int));
    v->sum_rejection_of_truth_fspt_score = calloc(classes, sizeof(float));
    v->sum_acceptance_of_truth_fspt_score = calloc(classes, sizeof(float));

    return v;
}

static void free_validation_data(validation_data *v) {
    /* Names */
    free_ptrs((void **) v->names, v->classes);
    /* True yolo detection */
    free(v->n_true_detection);
    free(v->sum_true_detection_iou);
    free(v->n_true_detection_rejection);
    free(v->n_true_detection_acceptance);
    free(v->sum_true_detection_rejection_fspt_score);
    free(v->sum_true_detection_acceptance_fspt_score);
    /* Wrong class yolo detection */
    free_ptrs((void **) v->n_wrong_class_detection, v->classes);
    free_ptrs((void **) v->sum_wrong_class_detection_iou, v->classes);
    free_ptrs((void **) v->n_wrong_class_rejection, v->classes);
    free_ptrs((void **) v->n_wrong_class_acceptance, v->classes);
    free_ptrs((void **) v->sum_wrong_class_rejection_fspt_score, v->classes);
    free_ptrs((void **) v->sum_wrong_class_acceptance_fspt_score, v->classes);
    /* False yolo detection */
    free(v->n_false_detection);
    free(v->n_false_detection_rejection);
    free(v->n_false_detection_acceptance);
    free(v->sum_false_detection_rejection_fspt_score);
    free(v->sum_false_detection_acceptance_fspt_score);
    /* No yolo detection */
    free(v->n_no_detection);
    free(v->sum_no_detection_iou);
    /* Fspt on truth */
    free(v->n_truth);
    free(v->n_rejection_of_truth);
    free(v->n_acceptance_of_truth);
    free(v->sum_rejection_of_truth_fspt_score);
    free(v->sum_acceptance_of_truth_fspt_score);
    /* Free validation data */
    free(v);
}

static void print_stats(char *datacfg, char *cfgfile, char *weightfile,
        char *outfile, char *export_score_base) {
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);

    FILE *outstream = outfile ? fopen(outfile, "w") : stderr;
    assert(outstream);

    list *fspt_layers = get_network_layers_by_type(net, FSPT);
    while (fspt_layers->size > 0) {
        layer *l = (layer *) list_pop(fspt_layers);
        for (int i = 0; i < l->classes; ++i) {
            fspt_t *fspt = l->fspts[i];
            fspt_stats *stats = get_fspt_stats(fspt, 0, NULL, 1);
            char buf[256] = {0};
            sprintf(buf, "%s class %s", l->ref, names[i]);
            print_fspt_criterion_args(outstream, fspt->c_args, buf);
            print_fspt_score_args(outstream, fspt->s_args, NULL);
            print_fspt_stats(outstream, stats, NULL);
            if (export_score_base) {
                char data_file[256] = {0};
                sprintf(data_file, "%s_%s_%s.txt",
                        export_score_base, l->ref, names[i]);
                FILE *data_stream = fopen(data_file, "w");
                assert(data_stream);
                export_score_data(data_stream, stats);
                fclose(data_stream);
            }
            free_fspt_stats(stats);
        }
    }
    if (outstream != stderr) fclose(outstream);
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
        char *outfile, char *save_weights_file, int *gpus, int ngpus,
        int clear, int refit, int ordered,
        int start, int end, int one_thread, int merge, int auto_only,
        int only_fit,
        int only_score, int print_stats_val, criterion_args **extern_c_args,
        score_args **extern_s_args) {
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.txt");
    char *backup_directory = option_find_str(options, "backup", "backup/");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    int n_nets = MAX(ngpus, 1);
    network **nets = calloc(n_nets, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < n_nets; ++i) {
        srand(seed);
#ifdef GPU
        if (ngpus > 0)
            cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= n_nets;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * n_nets;
    data train, buffer;

    list *tmp_fspt_layers = get_network_layers_by_type(net, FSPT);
    if (tmp_fspt_layers->size == 0)
        error("The net must have fspt layers.");
    layer fspt = *(layer *) tmp_fspt_layers->front->val;
    int classes = fspt.classes;
    int same_c_args = 1;
    int same_s_args = 1;
    while (tmp_fspt_layers->size > 0) {
        layer *l = (layer *) list_pop(tmp_fspt_layers);
        for (int i = 0; i < l->classes; ++i) {
            same_c_args &= compare_criterion_args(&l->fspt_criterion_args,
                    l->fspts[i]->c_args);
            same_s_args &= compare_score_args(&l->fspt_score_args,
                    l->fspts[i]->s_args);
        }
    }
    free_list(tmp_fspt_layers);

    if (auto_only) {
        only_fit = !same_c_args;
        merge |= only_fit;
        only_score = same_c_args && !same_s_args;
        if (same_c_args && same_s_args) {
            char buff[256];
            if (save_weights_file) {
                fprintf(stderr, "Save weights in %s.\n", save_weights_file);
                save_weights(net, save_weights_file);
            } else {
                sprintf(buff, "%s/%s_final.weights", backup_directory, base);
                fprintf(stderr, "Save weights in %s.\n", buff);
                save_weights(net, buff);
            }
            fprintf(stderr, "End of FSPT training without refit.\n");
            for (i = 0; i < n_nets; ++i) {
                free_network(nets[i]);
            }
            free(nets);
            free(names);
            return;
        }
    }

    if (only_fit) { 
        fprintf(stderr, "Only fit FSPTs...\n");
        fit_fspts(net, classes, refit, one_thread, merge);
    } else if (only_score) {
        fprintf(stderr, "Only score FSPTs...\n");
        score_fspts(net, classes, one_thread);
    } else {
        list *plist = get_paths(train_images);
        char **paths = (char **)list_to_array(plist);

        load_args args = get_base_args(net);
        args.coords = fspt.coords;
        args.paths = paths;
        args.n = imgs;
        args.classes = classes;
        args.jitter = fspt.jitter;
        args.num_boxes = fspt.max_boxes;
        args.d = &buffer;
        args.type = DETECTION_DATA;
        args.threads = N_CORES;
        args.ordered = ordered;
        args.beg = (0 <= start && start < plist->size) ? start : 0;
        args.m = (end && args.beg < end && end < plist->size) ?
            end : plist->size;

        pthread_t load_thread = load_data(args);
        double time;
        if (ordered) {
            net->max_batches = args.m / imgs;
        }
        while (get_current_batch(net) < net->max_batches) {
            time=what_time_is_it_now();
            pthread_join(load_thread, 0);
            train = buffer;
            args.beg = *net->seen;
            load_thread = load_data(args);
            fprintf(stderr, "Loaded: %lf seconds\n",
                    what_time_is_it_now()-time);
            time=what_time_is_it_now();
#ifdef GPU
            if(n_nets == 1){
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
        if(n_nets != 1) sync_nets(nets, n_nets, 0);
#endif
        fspt_layers_set_samples(net, refit, merge);
        merge = 0;
        char buff[256];
        sprintf(buff, "%s/%s_data_extraction.weights", backup_directory, base);
        save_weights(net, buff);
        fprintf(stderr, "Data extraction done. Fitting FSPTs...\n");
        fit_fspts(net, classes, refit, one_thread, merge);
        free_ptrs((void **)paths, plist->size);
        free_list(plist);
    } // end if (!only_fit && !only_score)
    char buff[256];
    if (save_weights_file) {
        fprintf(stderr, "Save weights in %s.\n", save_weights_file);
        save_weights(net, save_weights_file);
    } else {
        sprintf(buff, "%s/%s_final.weights", backup_directory, base);
        fprintf(stderr, "Save weights in %s.\n", buff);
        save_weights(net, buff);
    }
    fprintf(stderr, "End of FSPT training\n");
    list *fspt_layers = get_network_layers_by_type(net, FSPT);
    layer **fspt_layers_array = (layer **) list_to_array(fspt_layers);
    if (extern_c_args) {
        *extern_c_args = calloc(fspt_layers->size, sizeof(criterion_args));
        for (int k = 0; k < fspt_layers->size; ++k) {
            layer *l = fspt_layers_array[k];
            (*extern_c_args)[k] = *l->fspts[0]->c_args;
        }
    }
    if (extern_s_args) {
        *extern_s_args = calloc(fspt_layers->size, sizeof(score_args));
        for (int k = 0; k < fspt_layers->size; ++k) {
            layer *l = fspt_layers_array[k];
            (*extern_s_args)[k] = *l->fspts[0]->s_args;
            (*extern_s_args)[k].score_vol_n_array = NULL;
        }
    }
    if (print_stats_val) {
        FILE *outstream = outfile ? fopen(outfile, "w") : stderr;
        assert(outstream);
        for (int k = 0; k < fspt_layers->size; ++k) {
            layer *l = fspt_layers_array[k];
            for (int i = 0; i < l->classes; ++i) {
                fspt_t *fspt = l->fspts[i];
                fspt_stats *stats = get_fspt_stats(fspt, 0, NULL, 1);
                char buf[256] = {0};
                sprintf(buf, "%s class %s", l->ref, names[i]);
                print_fspt_criterion_args(outstream, fspt->c_args,
                        buf);
                print_fspt_score_args(outstream, fspt->s_args, NULL);
                print_fspt_stats(outstream, stats, NULL);
                free_fspt_stats(stats);
            }
        }
        if (outstream != stderr) fclose(outstream);
    }
    for (i = 0; i < n_nets; ++i) {
        free_network(nets[i]);
    }
    free(fspt_layers_array);
    free_list(fspt_layers);
    free(nets);
    free(names);
}

typedef struct valid_args {
    network *net;
    float yolo_thresh;
    float fspt_thresh;
    float hier_thresh;
    int *map;
    int classes;
    float nms;
    validation_data *val_data;
} valid_args;

static void *validate_thread(void *ptr) {
    valid_args args = *(valid_args *)ptr;
    network *net = args.net;
    int w = net->w;
    int h = net->h;
    float yolo_thresh = args.yolo_thresh;
    float fspt_thresh = args.fspt_thresh;
    float hier_thresh = args.hier_thresh;
    int *map = args.map;
    int classes = args.classes;
    float nms = args.nms;
    validation_data *val_data = args.val_data;
    /* FSPT boxes. */
    int *nboxes_fspt;
    detection **dets_fspt = get_network_fspt_boxes_batch(net, w, h,
            yolo_thresh, fspt_thresh, hier_thresh, map, 1, 0,
            &nboxes_fspt);
    if (nms) {
        for (int b = 0; b < net->batch; ++b)
            do_nms_suppression(dets_fspt[b], &nboxes_fspt[b], classes, nms);
    }
    /* FSPT truth boxes */
    int *nboxes_truth_fspt;
    detection **dets_truth_fspt =
        get_network_fspt_truth_boxes_batch(net, w, h,
                yolo_thresh, fspt_thresh, hier_thresh, map, 1,
                &nboxes_truth_fspt);

    for (int b = 0; b < net->batch; ++b) {
        update_validation_data(nboxes_fspt[b], dets_fspt[b],
                nboxes_truth_fspt[b], dets_truth_fspt[b], val_data);
        if (nboxes_fspt[b]) free_detections(dets_fspt[b], nboxes_fspt[b]);
        if (nboxes_truth_fspt[b])
            free_detections(dets_truth_fspt[b], nboxes_truth_fspt[b]);
    }
    free(dets_fspt);
    free(dets_truth_fspt);
    free(nboxes_fspt);
    free(nboxes_truth_fspt);
    free(ptr);
    return NULL;
}

static pthread_t validate_in_thread(network *net, float yolo_thresh,
        float fspt_thresh, float hier_thresh, int *map, int classes, 
        float nms, validation_data *val_data) {
    pthread_t thread;
    valid_args *ptr = calloc(1, sizeof(valid_args));
    ptr->net = net;
    ptr->yolo_thresh = yolo_thresh;
    ptr->fspt_thresh = fspt_thresh;
    ptr->hier_thresh = hier_thresh;
    ptr->map = map;
    ptr->classes = classes;
    ptr->nms = nms;
    ptr->val_data = val_data;
    if (pthread_create(&thread, 0, validate_thread, ptr))
        error("Thread creation failed");
    return thread;
}

static void validate_fspt(char *datacfg, char *cfgfile, char *weightfile,
        int n_yolo_thresh, float *yolo_threshs,
        int n_fspt_thresh, float *fspt_threshs, float hier_thresh,
        float iou_thresh, int ngpus, int *gpus, int ordered,
        int start, int end, 
        int print_stats_val, char *outfile, validation_data ***out_val_data) {
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/valid.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    int n_nets = MAX(ngpus, 1);
    network **nets = calloc(n_nets, sizeof(network));
    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < n_nets; ++i) {
        srand(seed);
#ifdef GPU
        if (ngpus > 0)
            cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, 1);
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * n_nets;
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

    double start_time = what_time_is_it_now();

    float nms = .45;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.classes = classes;
    args.jitter = 0;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = N_CORES;
    args.ordered = ordered;
    args.beg = (0 <= start && start < plist->size) ? start : 0;
    args.m = (end && args.beg < end && end < plist->size) ?
        end : plist->size;

    if (ordered) {
        net->max_batches = args.m / imgs;
    }

    validation_data **val_datas =
        calloc(n_yolo_thresh * n_fspt_thresh, sizeof(validation_data *));
    for (int i = 0; i < n_yolo_thresh; ++i) {
        for (int j = 0; j < n_fspt_thresh; ++j) {
            int index = i * n_fspt_thresh + j;
            val_datas[index] = allocate_validation_data(classes, names);
            validation_data *val_data = val_datas[index];
            val_data->n_images = net->max_batches * imgs;
            val_data->classes = classes;
            val_data->iou_thresh = iou_thresh;
            val_data->fspt_thresh = fspt_threshs[j];
        }
    }

    pthread_t load_thread = load_data(args);
    double time;
    while (get_current_batch(net) < net->max_batches) {
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        val = buffer;
        args.beg = *net->seen;
        load_thread = load_data(args);
        fprintf(stderr, "Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time=what_time_is_it_now();
#ifdef GPU
        if(n_nets == 1){
            validate_network_fspt(net, val);
        } else {
            validate_networks_fspt(nets, n_nets, val, 4);
        }
#else
        validate_network_fspt(net, val);
#endif
        i = get_current_batch(net);
        pthread_t *threads =
            calloc(n_yolo_thresh * n_fspt_thresh * n_nets, sizeof(pthread_t));
        for (int i = 0; i < n_yolo_thresh; ++i) {
            for (int j = 0; j < n_fspt_thresh; ++j) {
                int index = i * n_fspt_thresh + j;
                validation_data *val_data = val_datas[index];
                float fspt_thresh = fspt_threshs[j];
                float yolo_thresh = yolo_threshs[i];
                for (int k = 0; k < n_nets; ++k) {
                    // TODO: NOT THREAD SAFE IF N_NETS > 1
                    threads[index + k * n_yolo_thresh * n_fspt_thresh] =
                        validate_in_thread(nets[k], yolo_thresh,
                            fspt_thresh, hier_thresh, map, classes, nms,
                            val_data);
                }
            }
        }
        for (int i = 0; i < n_yolo_thresh * n_fspt_thresh; ++i) {
            pthread_join(threads[i], 0);
        }
        fprintf(stderr,
                "%ld: %lf seconds, %d images added to validation.\n",
                get_current_batch(net), what_time_is_it_now()-time,
                i*imgs);
        free(threads);
        free_data(val);
    }
    list *fspt_layers = get_network_layers_by_type(net, FSPT);
    layer **fspt_layers_array = (layer **) list_to_array(fspt_layers);
    fspt_stats **stats =
        calloc(fspt_layers->size * classes, sizeof(fspt_stats *));
    if (print_stats_val) {
        for (int i = 0; i < fspt_layers->size; ++i) {
            layer *l = fspt_layers_array[i];
            for (int j = 0; j < l->classes; ++j) {
                fprintf(stderr, "Computing fspt stats %s:%s.\n",
                        l->ref, names[j]);
                fspt_t *fspt = l->fspts[j];
                stats[i * classes + j] = get_fspt_stats(fspt, 0, NULL, 1);
            }
        }
    }
    for (int i = 0; i < n_yolo_thresh; ++i) {
        for (int j = 0; j < n_fspt_thresh; ++j) {
            int index = i * n_fspt_thresh + j;
            validation_data *val_data = val_datas[index];
            float fspt_thresh = fspt_threshs[j];
            float yolo_thresh = yolo_threshs[i];
            fprintf(stderr,"Yolo treshold is %g and fspt threshold is %g.\n",
                    yolo_thresh, fspt_thresh);
            FILE *outstream;
            char outfile2[256] = {0};
            if (outfile) {
                sprintf(outfile2,"%s_yolo_%g_fspt_%g",
                        outfile, yolo_thresh, fspt_thresh);
                outstream = fopen(outfile2, "w");
            } else {
                outstream = stderr;
            }
            assert(outstream);
            if (print_stats_val) {
                fprintf(stderr, "Print stats...\n");
                for (int k = 0; k < fspt_layers->size; ++k) {
                    layer *l = fspt_layers_array[k];
                    for (int c = 0; c < l->classes; ++c) {
                        fspt_t *fspt = l->fspts[c];
                        char buf[256] = {0};
                        sprintf(buf, "%s class %s", l->ref, names[c]);
                        print_fspt_criterion_args(outstream, fspt->c_args,
                                buf);
                        print_fspt_score_args(outstream, fspt->s_args, NULL);
                        print_fspt_stats(outstream, stats[k * classes + c],
                                NULL);
                    }
                }
            }
            fprintf(stderr, "Print validation...\n");
            print_validation_data(outstream, val_data, 0, "VALIDATION RESULT");
            if (!out_val_data && val_data) {
                free_validation_data(val_data);
            }
            if (outstream != stderr) fclose(outstream);
        }
    }
    for (int k = 0; k < fspt_layers->size; ++k) {
        for (int c = 0; c < classes; ++c) {
            if (stats[k * classes + c]) {
                free_fspt_stats(stats[k * classes + c]);
            }
        }
    }
    free(fspt_layers_array);
    free_list(fspt_layers);
    free(stats);
    if (!out_val_data)
        free(val_datas);
    else
        *out_val_data = val_datas;

    free_list(options);
    free_ptrs((void **) names, classes);
    for (int i = 0; i < n_nets; ++i) free_network(nets[i]);
    free(nets);
    free_ptrs((void **) paths, plist->size);
    free_list(plist);

    long t = 1000l * (what_time_is_it_now() - start_time);
    fprintf(stderr, "Total Validation Time : %ldh %ldm %lds %ldms.\n",
            t / 1000 / 60 / 60,
            t / 1000 / 60 % 60,
            t / 1000 % 60,
            t % 1000);
}

static double validation_score(const validation_data *v_positif,
        const validation_data *v_negatif) {
    float false_positif_p =
        safe_divd(v_positif->tot_n_true_detection_rejection,
                v_positif->tot_n_true_detection);
    float true_negatif_p = safe_divd(v_negatif->tot_n_rejection_of_truth,
            v_negatif->tot_n_truth);
    return true_negatif_p - false_positif_p;
}

typedef struct validation_cfg {
    float yolo_thresh;
    float fspt_thresh;
    validation_data *val_data_positif;
    validation_data *val_data_negatif;
    char *cfgfile;
    char *outfile_fit;
    char *outfile_val_positif;
    char *outfile_val_negatif;
    char *weightfile;
    int n_fspt_layers;
    criterion_args *c_args;
    score_args *s_args;
    int *n_input_layers;
    int **input_layers;
    float score;
} validation_cfg;

static void free_validation_cfg(validation_cfg v) {
    //TODO delete next line.
    return;
    free_validation_data(v.val_data_positif);
    free_validation_data(v.val_data_negatif);
    free(v.cfgfile);
    free(v.outfile_fit);
    free(v.outfile_val_positif);
    free(v.outfile_val_negatif);
    free(v.weightfile);
    free(v.c_args);
    free(v.s_args);
    free_ptrs((void **)v.input_layers, v.n_fspt_layers);
    free(v.n_input_layers);
}

static void print_validation_cfg(FILE *stream, validation_cfg *v, char *title){
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
    if (!v) {
        fprintf(stream, "No validation configuration.\n\n");
        return;
    }

    fprintf(stream, "\
┌────────────────────────────────────────────────────────────────────────────────────────────┐\n\
│                                VALIDATION CONFIGURATION                                    │\n\
├─────────────────────────────┬──────────────────────────────────────────────────────────────┤\n\
│                 yolo_thresh │"FLT_FORMAT"                                                  │\n\
│                 fspt_thresh │"FLT_FORMAT"                                                  │\n\
│          configuration file │%-62s│\n\
│                    fit file │%-62s│\n\
│  validation positif outfile │%-62s│\n\
│  validation negatif outfile │%-62s│\n\
│                 weight file │%-62s|\n\
│            validation score │"FLT_FORMAT"                                                  │\n\
└─────────────────────────────┴──────────────────────────────────────────────────────────────┘\n\n",
        v->yolo_thresh, v->fspt_thresh,
        v->cfgfile, v->outfile_fit, v->outfile_val_positif,
        v->outfile_val_negatif, v->weightfile, v->score
        );

    fprintf(stream, "Validation data positif resume :\n");
    print_validation_data(stream, v->val_data_positif, 1, NULL);
    fprintf(stream, "Validation data negatif resume :\n");
    print_validation_data(stream, v->val_data_negatif, 1, NULL);
    print_fspt_criterion_args(stream, v->c_args, NULL);
    print_fspt_score_args(stream, v->s_args, NULL);
}

static int cmp_val_cfg(const void *p1, const void *p2) {
    validation_cfg v1 = *(validation_cfg *)p1;
    validation_cfg v2 = *(validation_cfg *)p2;
    if (v1.score > v2.score)
        return -1;
    else
        return v1.score < v2.score;
}

static void validate_multiple_cfg(char *datacfg_positif, char *datacfg_negatif,
        int n_cfg, char **cfgfiles,
        char *weightfile, char *save_weightfile,
        int n_yolo_threshs, float *yolo_threshs,
        int n_fspt_threshs, float *fspt_threshs, float hier_thresh,
        float iou_thresh, int ngpus, int *gpus, int ordered,
        int start, int end, int one_thread, int auto_only,
        int print_stats_val, char *outfile) {

    validation_cfg *val_cfgs =
        calloc(n_cfg * n_yolo_threshs * n_fspt_threshs, sizeof(validation_cfg));

    assert(outfile);
    assert(save_weightfile);
    // local copy of the global variable gpu_index.
    int old_gpu_index = gpu_index;

    for (int cfg = 0; cfg < n_cfg; ++cfg) {
        char *cfgfile = cfgfiles[cfg];
        fprintf(stderr, "\nConfiguration file number %d : %s.\n", cfg,
                cfgfile);
        gpu_index = -1; // no gpu allocation needed for this net.
        network *net = load_network(cfgfile, NULL, 0);
        gpu_index = old_gpu_index;
        char *similar_weightfile = weightfile;
        list *fspt_layers = get_network_layers_by_type(net, FSPT);
        layer **fspt_layers_array = (layer **) list_to_array(fspt_layers);
        int n_fspt_layers = fspt_layers->size;
        if (auto_only) {
            /* Try to find a weightfile that was similar to avoid refitting. */
            for (int prev_cfg = 0; prev_cfg < cfg; ++prev_cfg) {
                validation_cfg val_cfg =
                    val_cfgs[prev_cfg * n_fspt_threshs * n_yolo_threshs];
                criterion_args *prev_c_args = val_cfg.c_args;
                score_args *prev_s_args = val_cfg.s_args;
                if (n_fspt_layers != val_cfg.n_fspt_layers) {
                    free_list(fspt_layers);
                    fprintf(stderr,
                            "Different number of fspt layer (%d:%d) than cfg %d.\n",
                            n_fspt_layers, val_cfg.n_fspt_layers,
                            prev_cfg);
                    continue;
                }
                int same_c_args = 1;
                int same_s_args = 1;
                for (int k = 0; k < n_fspt_layers; ++k) {
                    layer *l = fspt_layers_array[k];
                    same_c_args &=
                        l->inputs == val_cfg.n_input_layers[k];
                    if (!same_c_args) {
                        fprintf(stderr,
                                "Different number of input layers (%d:%d - %d) than cfg %d.\n",
                                l->inputs, val_cfg.n_input_layers[k], k,
                                prev_cfg);
                        break;
                    }
                    same_c_args &= equals_int_array(l->inputs,
                            l->input_layers, val_cfg.input_layers[k]);
                    if (!same_c_args) {
                        fprintf(stderr,
                                "Different input layers than cfg %d.\n",
                                prev_cfg);
                        break;
                    }
                    same_c_args &= compare_criterion_args(
                            &l->fspt_criterion_args, prev_c_args + k);
                    if (!same_c_args) {
                        fprintf(stderr,
                                "Different criterion args than cfg %d.\n",
                                prev_cfg);
                        print_fspt_criterion_args(stderr, &l->fspt_criterion_args, NULL);
                        print_fspt_criterion_args(stderr, prev_c_args + k, NULL);
                        break;
                    }
                    same_s_args &= compare_score_args(&l->fspt_score_args,
                            prev_s_args + k);
                    if (!same_s_args) {
                        fprintf(stderr,
                                "Different score args than cfg %d.\n",
                                prev_cfg);
                        break;
                    }
                }
                if (same_c_args) {
                    fprintf(stderr,
                            "Same criterion args than cfg %d.\n", prev_cfg);
                    similar_weightfile = val_cfg.weightfile;
                    if (same_s_args) {
                        fprintf(stderr,
                                "Same score args than cfg %d.\n", prev_cfg);
                        break;
                    }
                }
            }
        }
        /* Refit */
        char *outfile_fit = calloc(256, sizeof(char));
        sprintf(outfile_fit, "%s_cfg%d_fit", outfile, cfg);
        char *save_weightfile2 = calloc(256, sizeof(char));
        sprintf(save_weightfile2, "%s_cfg%d", save_weightfile, cfg);
        criterion_args *c_args;
        score_args *s_args;
        train_fspt(datacfg_positif, cfgfile, similar_weightfile, outfile_fit,
                save_weightfile2, gpus, ngpus, 1, 1, ordered, start,
                end, one_thread, 0,
                (auto_only && (similar_weightfile != weightfile)), 0, 0,
                print_stats_val, &c_args,
                &s_args);

        char *outfile_val_positif = calloc(256, sizeof(char));
        sprintf(outfile_val_positif, "%s_cfg%d_val_positif", outfile, cfg);
        char *outfile_val_negatif = calloc(256, sizeof(char));
        sprintf(outfile_val_negatif, "%s_cfg%d_val_negatif", outfile, cfg);
        validation_data **val_datas_positif;
        validation_data **val_datas_negatif;
        validate_fspt(datacfg_positif, cfgfile, save_weightfile2,
                n_yolo_threshs,
                yolo_threshs, n_fspt_threshs, fspt_threshs, hier_thresh,
                iou_thresh, ngpus, gpus, ordered, start, end, print_stats_val,
                outfile_val_positif, &val_datas_positif);
        validate_fspt(datacfg_negatif, cfgfile, save_weightfile2,
                n_yolo_threshs,
                yolo_threshs, n_fspt_threshs, fspt_threshs, hier_thresh,
                iou_thresh, ngpus, gpus, ordered, start, end, print_stats_val,
                outfile_val_negatif, &val_datas_negatif);

        fprintf(stderr, "Updata validation configuration %d.\n", cfg);

        for (int i = 0; i < n_yolo_threshs; ++i) {
            for (int j = 0; j < n_fspt_threshs; ++j) {
                int index = i * n_fspt_threshs + j;
                validation_data *val_data_positif = val_datas_positif[index];
                validation_data *val_data_negatif = val_datas_negatif[index];
                float fspt_thresh = fspt_threshs[j];
                float yolo_thresh = yolo_threshs[i];
                validation_cfg val_cfg = {0};
                val_cfg.yolo_thresh = yolo_thresh;
                val_cfg.fspt_thresh = fspt_thresh;
                val_cfg.val_data_positif = val_data_positif;
                val_cfg.val_data_negatif = val_data_negatif;
                val_cfg.cfgfile = copy_string(cfgfile);
                val_cfg.outfile_fit = outfile_fit;
                val_cfg.outfile_val_positif = outfile_val_positif;
                val_cfg.outfile_val_negatif = outfile_val_negatif;
                val_cfg.weightfile = save_weightfile2;
                val_cfg.n_fspt_layers = n_fspt_layers;
                val_cfg.c_args = c_args;
                val_cfg.s_args = s_args;

                val_cfg.n_input_layers = calloc(n_fspt_layers, sizeof(int));
                val_cfg.input_layers = calloc(n_fspt_layers, sizeof(int *));
                for (int k = 0; k < n_fspt_layers; ++k) {
                    layer *l = fspt_layers_array[k];
                    val_cfg.n_input_layers[k] = l->inputs;
                    val_cfg.input_layers[k] =
                        copy_int_array(l->inputs, l->input_layers);
                }

                val_cfg.score =
                    validation_score(val_data_positif, val_data_negatif);

                int bigindex = cfg * n_fspt_threshs * n_yolo_threshs + index;
                val_cfgs[bigindex] = val_cfg;
            }
        }
        free(fspt_layers_array);
        free_list(fspt_layers);
        /* Print resume for this configuration */
        int beg = cfg * n_fspt_threshs * n_yolo_threshs;
        qsort(val_cfgs + beg, n_yolo_threshs * n_fspt_threshs,
                sizeof(validation_cfg), cmp_val_cfg);
        char *outfile_resume = calloc(256, sizeof(char));
        sprintf(outfile_resume, "%s_cfg%d_resume", outfile, cfg);
        FILE *f = fopen(outfile_resume, "w");
        for (int i = beg; i < beg + n_yolo_threshs * n_fspt_threshs; ++i) { 
            char title[512] = {0};
            sprintf(title,
                    "Configuration number %d - fspt thresh %f, yolo thresh %f",
                    i - beg, val_cfgs[i].fspt_thresh, val_cfgs[i].yolo_thresh);
            print_validation_cfg(f, val_cfgs + i, title);       
        }
        fclose(f);
        /* Free configuration */
        fprintf(stderr, "Free network configuration %d.\n", cfg);

        gpu_index = -1; // no gpu space to free in this net.
        free_network(net);
        gpu_index = old_gpu_index;
    }
    /* Resume */
    qsort(val_cfgs, n_cfg * n_yolo_threshs * n_fspt_threshs,
            sizeof(validation_cfg), cmp_val_cfg);
    FILE *f = fopen(outfile, "w");
    for (int i = 0; i < n_cfg * n_yolo_threshs * n_fspt_threshs; ++i) { 
        char title[256] = {0};
        sprintf(title, "Configuration number %d", i);
        print_validation_cfg(f, val_cfgs + i, title);       
    }
    fclose(f);
    fprintf(stderr, "Free validation data.\n");
    for (int i = 0; i < n_cfg * n_yolo_threshs * n_fspt_threshs; ++i) { 
        free_validation_cfg(val_cfgs[i]);
    }
    free(val_cfgs);
}


void run_fspt(int argc, char **argv) {
    if(argc < 4) {
        fprintf(stderr,
"usage: %s %s <train/test/valid> <datacfg> <netcfg> [weights]\n\
                                                    [inputfile] [options]\n\
   or: %s %s <valid_multiple> <netcfgs> [weights] -pos <negconf>\n\
                                                 -neg <posconf> [options]\n\
With :\n\
    train -> train fspts on an already trained yolo network.\n\
    test  -> test fspt predictions.\n\
    valid -> validate fspt.\n\
    valid_multiple -> validate multiple fspt configurations.\n\
    stats -> print statistics of the fspts.\n\
And :\n\
    <datacfg>   -> path to the data configuration file.\n\
    <negconf>   -> path to the data configuration file for negatif validation.\n\
    <posconf>   -> path to the data configuration file for positive validation.\n\
    <netcfg>    -> path to the network configuration file.\n\
    <netcfgs>   -> comma separated paths to the network configuration files.\n\
    [weights]   -> path to a weightfile corresponding to the netcfg file.\n\
                   (optional)\n\
    [inputfile] -> path to a file with list of test images.\n\
                   only for commande test. (optional)\n\
Options are :\n\
    -yolo_thresh -> comma separated yolo detection threashold. default 0.5.\n\
    -fspt_thresh -> comma separated fspt rejection threshold. default 0.5.\n\
    -hier        -> unused.\n\
    -gpus        -> coma separated list of gpus.\n\
    -out         -> ouput file for prints.\n\
    -save_weights_file -> ouput file to save weights.\n\
    -export      -> ouput file for score raw data.\n\
    -clear       -> if set, the training number of seen images is reset.\n\
    -refit       -> if set, the fspts are refitted if they already exist.\n\
    -ordered     -> if set, the data are selected sequentialy and not randomly.\n\
    -start       -> indicate the line of the first (included and\n\
                    starting from 0) image link. Default 0.\n\
    -end         -> indicate the line of the last (excluded and\n\
                    starting from 0) image link. Default last link.\n\
    -merge       -> if set, newly extracted data are merged to existing.\n\
    -auto_only   -> sets automatically only_fit and only_score. The extarcted\n\
                    data are the ones in the weight file.\n\
    -only_fit    -> if set, don't extract new data. implies -merge.\n\
    -only_score  -> if set, only scores the fspts.\n\
    -one_thread  -> if set, the fspts are fitted in only one thread instead of\n\
                    one thread per fspt.\n\
    -fullscreen  -> unused.\n\
    -print_stats -> if set, print the statistics of the fspts after training.\n",
                argv[0], argv[1], argv[0], argv[1]);
        return;
    }

    char *yolo_thresh_list = find_char_arg(argc, argv, "-yolo_thresh", ".5");
    char *fspt_thresh_list = find_char_arg(argc, argv, "-fspt_thresh", ".5");
    char *posconf = find_char_arg(argc, argv, "-pos", 0);
    char *negconf = find_char_arg(argc, argv, "-neg", 0);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    float iou_thresh = find_float_arg(argc, argv, "-iou", .5);
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *save_weights_file =
        find_char_arg(argc, argv, "-save_weights_file", 0);
    char *export_score_file = find_char_arg(argc, argv, "-export", 0);
    int clear = find_arg(argc, argv, "-clear");
    int refit_fspts = find_arg(argc, argv, "-refit");
    int ordered = find_arg(argc, argv, "-ordered");
    int start = find_int_arg(argc, argv, "-start", 0);
    int end = find_int_arg(argc, argv, "-end", 0);
    int one_thread = find_arg(argc, argv, "-one_thread");
    int auto_only = find_arg(argc, argv, "-auto_only");
    int only_fit = find_arg(argc, argv, "-only_fit");
    int only_score = find_arg(argc, argv, "-only_score");
    int merge = find_arg(argc, argv, "-merge") || only_fit;
    int print_stats_val = find_arg(argc, argv, "-print_stats");
    int fullscreen = find_arg(argc, argv, "-fullscreen");

    /* gpus */
    int *gpus = 0;
    int ngpus = 0;
    if(gpu_list){
        int len = strlen(gpu_list);
        ngpus = 1;
        for(int i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(int i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    /* fspt thresh */
    int len = strlen(fspt_thresh_list);
    int n_fspt_thresh = 1;
    for(int i = 0; i < len; ++i){
        if (fspt_thresh_list[i] == ',') ++n_fspt_thresh;
    }
    float *fspt_threshs = calloc(n_fspt_thresh, sizeof(float));
    for(int i = 0; i < n_fspt_thresh; ++i){
        fspt_threshs[i] = atof(fspt_thresh_list);
        fspt_thresh_list = strchr(fspt_thresh_list, ',') + 1;
    }
    /* yolo threshs */
    len = strlen(yolo_thresh_list);
    int n_yolo_thresh = 1;
    for(int i = 0; i < len; ++i){
        if (yolo_thresh_list[i] == ',') ++n_yolo_thresh;
    }
    float *yolo_threshs = calloc(n_yolo_thresh, sizeof(float));
    for(int i = 0; i < n_yolo_thresh; ++i){
        yolo_threshs[i] = atof(yolo_thresh_list);
        yolo_thresh_list = strchr(yolo_thresh_list, ',') + 1;
    }

    char *datacfg = NULL;
    char *cfg = NULL;
    char *weights = NULL;
    char *filename = NULL;
    int n_cfgs = 0;
    char **cfgs = NULL;
    if (strcmp(argv[2], "valid_multiple")) {
        datacfg = argv[3];
        cfg = argv[4];
        weights = (argc > 5) ? argv[5] : 0;
        filename = (argc > 6) ? argv[6]: 0;
    } else {
        assert(posconf && negconf);
        weights = (argc > 4) ? argv[4] : 0;
        filename = (argc > 5) ? argv[5]: 0;
        /* configuration files */
        char *cfg_list = argv[3];
        len = strlen(cfg_list);
        n_cfgs = 1;
        for(int i = 0; i < len; ++i){
            if (cfg_list[i] == ',') ++n_cfgs;
        }
        cfgs = calloc(n_cfgs, sizeof(char *));
        cfgs[0] = strtok(cfg_list, ",");
        for(int i = 1; i < n_cfgs; ++i){
            cfgs[i] = strtok(NULL, ",");
        }
    }
    if(0==strcmp(argv[2], "test"))
        test_fspt(datacfg, cfg, weights, filename, *yolo_threshs,
                *fspt_threshs, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train"))
        train_fspt(datacfg, cfg, weights, outfile, save_weights_file, gpus,
                ngpus, clear,
                refit_fspts, ordered, start, end, one_thread, merge, auto_only,
                only_fit,
                only_score, print_stats_val, NULL, NULL);
    else if(0==strcmp(argv[2], "valid"))
        validate_fspt(datacfg, cfg, weights, n_yolo_thresh, 
                yolo_threshs, n_fspt_thresh, fspt_threshs,
                hier_thresh, iou_thresh, ngpus, gpus, ordered, start, end,
                print_stats_val, outfile, NULL);
    else if(0==strcmp(argv[2], "valid_multiple"))
        validate_multiple_cfg(posconf, negconf, n_cfgs, cfgs, weights,
                save_weights_file, n_yolo_thresh, 
                yolo_threshs, n_fspt_thresh, fspt_threshs,
                hier_thresh, iou_thresh, ngpus, gpus, ordered, start, end,
                one_thread, auto_only,
                print_stats_val, outfile);
    else if (0 == strcmp(argv[2], "stats"))
        print_stats(datacfg, cfg, weights, outfile, export_score_file);

    if (gpus) free(gpus);
    free(fspt_threshs);
    free(yolo_threshs);
}

#undef FLT_FORMAT
#undef INT_FORMAT
