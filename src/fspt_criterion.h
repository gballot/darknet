#ifndef GINI_H
#define GINI_H

#include <stddef.h>

#include "fspt.h"

typedef struct criterion_args {
    /* messages to change fitting behaviour */
    int merge_nodes;
    /* messages between fspt_fit and all the criterion functions */
    fspt_t *fspt;
    fspt_node *node;
    int max_depth;
    size_t count_max_depth_hit;
    int min_samples;
    size_t count_min_samples_hit;
    double min_volume_p;
    size_t count_min_volume_p_hit;
    double min_length_p;
    size_t count_min_length_p_hit;
    size_t count_max_count_hit;
    size_t count_no_sample_hit;
    int best_index;
    float best_split;
    int forbidden_split;
    int increment_count;
    int end_of_fitting;
    /* messages for gini_criterion */
    float max_tries_p;
    float max_features_p;
    double gini_gain_thresh;
    int max_consecutive_gain_violations;
    int middle_split;
} criterion_args;


extern criterion_args *load_criterion_args_file(FILE *fp, int *succ);

extern void save_criterion_args_file(FILE *fp, criterion_args *c, int *succ);

/**
 * Convert a string representing the name of a criterion function
 * to a pointer to this function.
 *
 * \param s the string name of hte criterion function.
 * \return A pointer to this criterion function if the name exists or
 *         NULL.
 */
extern criterion_func string_to_fspt_criterion(char *s);

/**
 * The gini criterion function.
 * This criterion function from Toward Safe Machine Learning paper returns
 * the splits that maximize the gain in the gini index before and after the
 * potential split.
 *
 * \param args Input/Output parameter.
 */
extern void gini_criterion(criterion_args *args);

/**
 * Prints a criterion_args to a file.
 *
 * \param stream The output stream.
 * \param a The criterion_args.
 * \param title An optional title.
 */
extern void print_fspt_criterion_args(FILE *stream, criterion_args *a,
        char *title);

#ifdef DEBUG
extern void hist(size_t n, size_t step, const float *X, float lower_bond,
                 size_t *n_bins, size_t *cdf, float *bins);
#endif

#endif /* GINI_H */
