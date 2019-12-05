#ifndef FSPT_SCORE_H
#define FSPT_SCORE_H

#include "fspt.h"

typedef struct density_normalize_args {
    double tau;
    int verification_passed;
} density_normalize_args;

typedef struct score_args {
    /* messages to change fitting behaviour */
    int score_during_fit;
    /* messages for all score functions */
    fspt_t *fspt;
    fspt_node *node;
    int discover;
    int need_normalize;
    int normalize_pass;
    size_t n_leaves;
    score_vol_n *score_vol_n_array; // size n_leaves.
    /* messages for euristic score */
    int compute_euristic_hyperparam;
    float euristic_hyperparam;
    /* message for density score */
    int exponential_normalization;
    double calibration_score;
    double calibration_n_samples_p;
    double calibration_volume_p;
    float calibration_feat_length_p;
    double volume_penalization;
    double calibration_tau;
    /* message for auto density score */
    int compute_norm_args;
    double samples_p;
    double verify_density_thresh;
    double verify_n_nodes_p_thresh;
    double auto_calibration_score;
    density_normalize_args norm_args;
} score_args;


extern score_args *load_score_args_file(FILE *fp, int *succ);

extern void save_score_args_file(FILE *fp, score_args *s, int *succ);
/**
 * maps the string names of the functions to the score functions.
 *
 * \param s The string name of a score function.
 * \return the corresponding score function if it exists or NULL.
 */
extern score_func string_to_fspt_score(char *s);

/**
 * Prints a score_args to a file.
 *
 * \param stream The output stream.
 * \param a The score_args.
 * \param title An optional title.
 */
extern void print_fspt_score_args(FILE *stream, score_args *a, char *title);

/**
 * A score function called "density".
 * This score is the density of samples in the node normalized by the
 * density of the whole fspt.
 *
 * \param args The score arguments. Should at least contain the node and
 *             the fspt.
 */
extern double density_score(score_args *args);

/**
 * A score function called "auto_density".
 * The raw score is the density of samples in the node normalized by the
 * density of the whole fspt.
 * Then it is normalized by some automatic technics.
 *
 * \param args The score arguments. Should at least contain the node and
 *             the fspt.
 */
extern double auto_normalized_density_score(score_args *args);

/**
 * An euristic score function called "euristic".
 * This score is :
 * \sum f_i \frac{ R^+ } { R^+ + E \frac{\Delta I(R)} {\Delta I} }
 * Where f_i is the feature importance of feature i, R^+ is the number
 * of training samples it the node, \Delta I(R) is the differance
 * between the feature limits of feature i in the node, \Delta I is the
 * differance between the feature limits of feature i in the fspt, 
 * and E is the ratio between training samples in the tree and number of
 * feature.
 *
 * \param args The score arguments. Should at least contain the node and
 *             the fspt.
 */
extern double euristic_score(score_args *args);

#endif /* FSPT_SCORE_H */
