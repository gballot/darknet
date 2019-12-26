#ifndef GINI_H
#define GINI_H

#include <stddef.h>

#include "fspt.h"

typedef enum CRITERION_FUNCTION {
    UNKNOWN_CRITERION_FUNC = 0,
    GINI = 1
} CRITERION_FUNCTION;

typedef struct {
    size_t count_max_depth_hit;
    size_t count_min_samples_hit;
    size_t count_min_volume_p_hit;
    size_t count_min_length_p_hit;
} forbidden_split_cause;

typedef enum unf_test_level {
    NO_TEST_UNIFORMITY = 0,
    MIXED_TEST_UNIFORMITY = 1,
    ALLWAYS_TEST_UNIFORMITY = 2
} UNF_TEST_LEVEL;

typedef struct criterion_args {
    /* messages to change fitting behaviour */
    CRITERION_FUNCTION criterion_function;
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
    size_t count_uniformity_hit;
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
    int multi_threads;
    UNF_TEST_LEVEL uniformity_test_level;
    float unf_alpha;
} criterion_args;


extern criterion_args *load_criterion_args_file(FILE *fp, int *succ);

extern void save_criterion_args_file(FILE *fp, criterion_args *c, int *succ);

extern int respect_min_lenght_p(int n_features, const float* fspt_lim,
        const float *node_lim, double min_length_p);

extern void determine_cause(int n, forbidden_split_cause *causes,
        criterion_args *args);

/**
 * maps the string names of the functions to the criterion functions
 * number.
 *
 * \param s The string name of a criterion function.
 * \return the corresponding criterion function number if it exists or 0.
 */
extern CRITERION_FUNCTION string_to_criterion_function_number(char *s);

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
 * Compares two criterion arguments.
 *
 * \param c1 The first criterion argument.
 * \param c2 The second criterion argument.
 * \return 1 if the arguments are equal. 0 otherwise.
 */
extern int compare_criterion_args(const criterion_args *c1,
        const criterion_args *c2);

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
