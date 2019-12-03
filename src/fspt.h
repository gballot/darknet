/**
 * fspt.c implements the feature of the Feature Space Partitioning Tree.
 *
 * The FSPT identifies the subspaces in the feature space that contains too
 * few data to trust the prediction of a learning model.
 * See : Toward Safe Machine Learning, by Arvind Easwaran and Xiaozhe Gu.
 * \author Gabriel Ballot
 */

#ifndef FSPT_H
#define FSPT_H

#include <stddef.h>
#include <stdio.h>

#include "list.h"



typedef enum {LEAF, INNER} FSTP_NODE_TYPE;
typedef enum {PRE_ORDER, IN_ORDER, POST_ORDER} FSPT_TRAVERSAL;
typedef enum {SPLIT = 0, UNKNOW, MAX_DEPTH, MIN_SAMPLES, MIN_VOLUME,
    MIN_LENGTH, MAX_COUNT, NO_SAMPLE} NON_SPLIT_CAUSE;


struct fspt_node;
struct fspt_t;
struct criterion_args;
struct score_args;
typedef void (*criterion_func) (struct criterion_args *args);
typedef double (*score_func) (struct score_args *args);

/**
 * Node of the FSPT.
 */
typedef struct fspt_node {
    FSTP_NODE_TYPE type;  // LEAF or INNER
    int n_features;
    struct fspt_t *fspt;   // the fspt that contains this node
    size_t n_empty;        // number of empty points (is a float)
    size_t n_samples;
    float *samples;     // training samples
    int split_feature;  // splits on feature SPLIT_FEATURE
    float split_value;   // go to right child if feature[i] > split_value
                         // to the left chil if feature[i] <= split_value.
    struct fspt_node *right;   // right child
    struct fspt_node *left;    // left child
    struct fspt_node *parent;  // the parent node
    int depth;
    double score;
    double volume;
    int count;          // keeps the successive violation of gain threshold
    NON_SPLIT_CAUSE cause;
} fspt_node;

/**
 * Feature Space Partitioning Tree.
 */
typedef struct fspt_t {
    int n_features;      // number of features
    const float *feature_limit;// size 2*n_feature:
                         //feature_limit[2*i] = min feature(i)
                         // feature_limite[2*i+1] = max feature(i)
    const float *feature_importance; // feature_importance of size n_feature
    size_t n_nodes;         // number of nodes
    size_t n_samples;       // number of training samples
    float *samples;     // training samples
    fspt_node *root;
    criterion_func criterion; // spliting criterion
    score_func score;    // score_function
    int depth;
    int count;          // keeps the successive violation of gain threshold
    double volume;      // total volume of the fspt
    struct criterion_args *c_args;
    struct score_args *s_args;
} fspt_t;


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

typedef struct score_args {
    /* messages to change fitting behaviour */
    int score_during_fit;
    /* messages for all score functions */
    fspt_t *fspt;
    fspt_node *node;
    int discover;
    int need_normalize;
    int normalize_pass;
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
} score_args;

typedef struct score_vol_n {
    double score;
    double volume_p;
    size_t n_samples;
    NON_SPLIT_CAUSE cause;
} score_vol_n;

typedef struct fspt_stats {
    /* Inputs */
    fspt_t *fspt;                 // Fspt related to this statistics.
    int n_thresh;                 // Number of thresholds.
    double *fspt_thresh;           // Score thresholds for some statistics.
    /* Volume statistics */
    double volume;                // Volume of the fspt.
    double leaves_volume;         // Volume of the leaves.
    double mean_volume;           // Mean volume of leaves.
    double min_volume_parameter;  // Value of the fitting parameter min_volume.
    double min_volume;            // Min volume of leaves.
    double max_volume;            // Max volume of leaves.
    double median_volume;         // Median volume of leaves.
    double first_quartile_volume; // First quartile volume of leaves.
    double third_quartile_volume; // Third quartile volume of leaves.
    double *volume_above_thresh;  // Size n_thresh. Sum volume of leaves with
                                  // score above each thresholds.
    double leaves_volume_p;         // Proportional volume of the leaves.
    double mean_volume_p;         // Proportional mean volume of leaves.
    double min_volume_p;          // Proportional min volume of leaves.
    double max_volume_p;          // Proportional max volume of leaves.
    double median_volume_p;       // Proportional median volume of leaves.
    double first_quartile_volume_p; // Proportional first quartile volume of
                                    // leaves.
    double third_quartile_volume_p; // Proportional third quartile volume of
                                    // leaves.
    double *volume_above_thresh_p;  // Size n_thresh. Proportion of the sum of
                                    // the volume of the leaves with score
                                    // above each thresholds.
    /* Number of samples statistics */
    size_t n_samples;                // Number of samples of the tree.
    int min_samples_param;        // Parameter of the minimum number of
                                  // sample per nodes.
    double mean_samples_leaves;    // Mean number of samples per leaves.
    size_t min_samples_leaves;       // Minimum number of samples per leaves.
    size_t max_samples_leaves;       // Maximum number of samples per leaves.
    double median_samples_leaves;    // Median number of samples per leaves.
    double first_quartile_samples_leaves; // First Q number of samples per leaves.
    double third_quartile_samples_leaves; // Thid Q number of samples per leaves.
    size_t *n_samples_above_thresh;  // Number of samples in leaves with score
                                  // above each thresholds.
    double mean_samples_leaves_p;    // Proportional mean number of samples per
                                    // leaves.
    double min_samples_leaves_p;     // Proportional minimum number of samples
                                    // per leaves.
    double max_samples_leaves_p;     // Proportional maximum number of samples
                                    // per leaves.
    double median_samples_leaves_p;  // Proportional median number of samples
                                    // per leaves.
    double first_quartile_samples_leaves_p; // Proportional first Q number of
                                           // samples per leaves.
    double third_quartile_samples_leaves_p; // Proportional thid Q number of
                                           // samples per leaves.
    double *n_samples_above_thresh_p;// Proportional number of samples in leaves
                                    // with score above each thresholds.
    /* Depth statistics */
    int max_depth;           // Max depth parameter of the tree.
    int depth;               // Depth of the tree.
    int min_depth_leaves;    // Min depth of the leaves.
    double mean_depth_leaves; // Mean depth of the leaves.
    double median_depth_leaves; // Median depth of the leaves.
    double first_quartile_depth_leaves;  // First Q of depth of leaves.
    double third_quartile_depth_leaves;  // Third Q of depth of leaves.
    double min_depth_leaves_p;    // Proportional min depth of the leaves.
    double mean_depth_leaves_p;   // Proportional mean depth of the leaves.
    double median_depth_leaves_p; // Proportional median depth of the leaves.
    double first_quartile_depth_leaves_p;  // Proportional first Q of depth of
                                          // leaves.
    double third_quartile_depth_leaves_p;  // Proportional third Q of depth of
                                          // leaves.
    double balanced_index;    // Score between 0 and 1. 0 if line tree,
                             // 1 if balanced. (1 - (2*depth-1)/n_nodes).
    size_t *n_nodes_by_depth;   // Size depth. Number of nodes by depth.
    double *n_nodes_by_depth_p; // Size depth. Number of nodes by depth divided
                                // by 2^(depth-1).
    /* Node type statistics */
    size_t n_leaves;       // Number of leaves.
    size_t n_inner;        // Number of inner nodes.
    double n_leaves_p;   // Proportional number of leaves.
    double n_inner_p;    // Proportional number of inner nodes.
    size_t *n_leaves_above_thresh;  // Size n_thresh. Number of nodes above each
                                 //thresholds.
    double *n_leaves_above_thresh_p;  // Size n_thresh. Proportional number of
                                     //nodes above each thresholds.
    score_vol_n *score_vol_n_array;
    /* Split statistics */
    int *split_features_count;    // Size n_features. Value at index i
                                  // is the number of split on feature i.
    double *split_features_count_p;  // Size n_features. Value at index i
                                    // is the proportional number of split on
                                  // feature i.
    double *min_split_values;      // Size n_features. Min split value by
                                  // features.
    double *max_split_values;      // Size n_features. Max split value by
                                  // features.
    double *mean_split_values;     // Size n_features. Mean split value by
                                  // features.
    double *median_split_values;   // Size n_features. Median split value by
                                  // features.
    double *first_quartile_split_values; // Size n_features. Fist Q split value
                                        // by features.
    double *third_quartile_split_values; // Size n_features. Third Q split value
                                        // by features.
    /* Score statistics */
    double mean_score;    // Mean score of leaves.
    double min_score;      // Min score of leaves.
    double max_score;     // Max score of leaves.
    double median_score;  // Median score of leaves.
    double first_quartile_score;  // Fist Q score of leaves.
    double third_quartile_score;  // Third Q score of leaves.
} fspt_stats;

/**
 * Builds an empty feature space partitioning tree.
 * 
 * \param n_features The number of features.
 * \param feature_limit values at index i and i+1 are respectively
 *                      the min and max of feature i.
 * \param feature_importance The static feature importance. Initialized at 1.
 *                           if NULL.
 * \param criterion The criterion to optimize.
 * \param score The score function for the leaves.
 * \return A pointer to the fspt_tree. Must be freed by the caller with
 *         a call to fspt_free(fspt_tree fspt).
 */
extern fspt_t *make_fspt(
        int n_features,
        const float *feature_limit,
        const float *feature_importance,
        criterion_func criterion,
        score_func score);

/**
 * Computes recursively the feature limit of the node `node`.
 *
 * \param node The node to compute the feature_limit.
 */
extern float *get_feature_limit(const fspt_node *node);

/**
 * Computes the total volume of the leaf nodes with score higher
 * than `thresh`.
 *
 * \param thresh The thresh to consider the nodes
 * \param fspt The fspt we want to compute volume.
 * \return The volume.
 */
extern double get_fspt_volume_score_above(float thresh, fspt_t *fspt);

/**
 * Gives the nodes containing each input X. The output parameter nodes
 * needs to be freed by the caller. But not the nodes themselves.
 *
 * \param n The number of test samples in X.
 * \param fspt The feature space partitioning tree.
 * \param X Size (n * fspt->n_features), containing the inputs to test.
 * \param nodes Output parameter that will be filled by the n nodes.
 *              Needs to be freed by the caller.
 */ 
extern void fspt_decision_func(size_t n, const fspt_t *fspt, const float *X,
                        fspt_node **nodes);

/**
 * Gives the score for each input X. The output parameter Y
 * needs to be freed by the caller.
 *
 * \param n The number of test samples in X.
 * \param fspt The feature space partitioning tree.
 * \param X Size (n * fspt->n_features), containing the inputs to test.
 * \param Y Output parameter that will be filled by the n predictions.
 *          needs to be freed by the caller.
 */ 
extern void fspt_predict(size_t n, const fspt_t *fspt, const float *X, float *Y);

/**
 * Fits the feature space partitioning tree to the data X.
 *
 * \param n_samples The number of samples in X.
 * \param X the samples to fit.
 * \param c_args Pointer to the criterion args.
 * \param c_args Pointer to the score args.
 * \param fspt The feature space partitioning tree.
 */
extern void fspt_fit(size_t n_samples, float *X, criterion_args *c_args,
        score_args *s_args, fspt_t *fspt);

/**
 * Recompute the score of the leaves without fitting.
 *
 * \param fspt The fspt.
 * \param s_args The new score arguments.
 */
extern void fspt_rescore(fspt_t *fspt, score_args *s_args);

/**
 * Save the fspt to a file.
 *
 * \param filename The path of the file to save.
 * \param fspt The feature space partitioning tree.
 * \param save_samples If true, the samples will be saved.
 * \param succ Output parameter. True if succes, false otherwise.
 */
extern void fspt_save(const char *filename, fspt_t fspt, int save_samples,
        int *succ);

/**
 * Save the fspt to an open file.
 *
 * \param fp A file pointer. Should be open and closed by the caller.
 * \param fspt The feature space partitioning tree.
 * \param save_samples If true, the samples will be saved.
 * \param succ Output parameter. True if succes, false otherwise.
 */
extern void fspt_save_file(FILE *fp, fspt_t fspt, int save_samples, int *succ);

/**
 * Load the fspt from a file.
 * You must have created the fspt with @see make_fspt(), because
 * some fields of the fspt are assumed to be already filled.
 * This function fills n_nodes, n_samples, depth, vol and root.
 *
 * \param filename The path of the file to save.
 * \param fspt The feature space partitioning tree parsed from config file.
 * \param load_samples If true, it will load the samples if they are
 *                     registered.
 * \param load_c_args If true, it will load the cirterion_args if it is
 *                    registered.
 * \param load_s_args If true, it will load the score_args if it is
 *                    registered.
 * \param load_root If true, it will load the root if it is registered.
 * \param succ Output parameter. True if succes, false otherwise.
 */
extern void fspt_load(const char *filename, fspt_t *fspt, int load_samples,
        int load_c_args, int load_s_args, int load_root, int *succ);

/**
 * Load the fspt from an open file.
 * You must have created the fspt with @see make_fspt(), because
 * some fields of the fspt are assumed to be already filled.
 * This function fills n_nodes, n_samples, depth, vol and root.
 *
 * \param fp A file pointer. Should be open and closed by the caller.
 * \param fspt The feature space partitioning tree parsed from config file.
 * \param load_samples If true, it will load the samples if they are
 *                     registered.
 * \param load_c_args If true, it will load the cirterion_args if it is
 *                    registered.
 * \param load_s_args If true, it will load the score_args if it is
 *                    registered.
 * \param load_root If true, it will load the root if it is registered.
 * \param succ Output parameter. True if succes, false otherwise.
 */
extern void fspt_load_file(FILE *fp, fspt_t *fspt, int load_samples,
        int load_c_args, int load_s_args, int load_root, int *succ);

/**
 * Friendly prints a fspt to the terminal.
 *
 * \param fspt The feature space partitioning tree.
 */
extern void print_fspt(fspt_t *fspt);

/**
 * Recursively frees fspt_nodes. Don't free the samples because they are
 * shared between all the tree. The samples are freed by free_fspt.
 *
 * \param node The root from where the nodes are freed.
 */
extern void free_fspt_nodes(fspt_node *node);

/**
 * Frees a fspt with call to free_fspt_nodes.
 * Frees the samples.
 *
 * \param fspt The fspt that will be freed.
 */
extern void free_fspt(fspt_t *fspt);

/**
 * Creates text file with column :
 * index, score, volume_p, mean_length_p, n_samples, n_samples_p, density
 *
 * \param stream The stream to write data.
 * \param s The fspt stats.
 */
extern void export_score_data(FILE *stream, fspt_stats *s);

/**
 * Prints the stats in a friendly format to stream.
 * @see get_fspt_stats().
 *
 * \param stream the output stream.
 * \param stats The fspt statistics.
 * \param title An optional title for the stats. can be NULL.
 */
extern void print_fspt_stats(FILE *stream, fspt_stats *stats, char *title);

/**
 * Creates a list with the nodes of a fspt. The traversal mode can be
 * customized. The caller must free the list.
 *
 * \param fspt The fspt/
 * \param traversal The mode of traversal @see FSPT_TRAVERSAL.
 * \return The list of all the nodes.
 */
extern list *fspt_nodes_to_list(fspt_t *fspt, FSPT_TRAVERSAL traversal);

/**
 * Extract a bunch of statistics about an fspt. @see fspt_stats.
 * The fspt_stats returned must be freed by the caller @see free_fspt_stats().
 * 
 * \param fspt The fspt to analyse.
 * \param n_thresh The number of thresholds in fspt_thresh or 0 for automatic
 *                 fspt_thresh. @see N_THRESH_STATS_FSPT.
 * \param fspt_thresh Array of thresholds for stats or NULL for automatic.
 */
extern fspt_stats *get_fspt_stats(fspt_t *fspt, int n_thresh,
        double *fspt_thresh);

/**
 * Frees the fspt_stats except the fspt himself.
 *
 * \param stats The fspt statistics to free.
 */
extern void free_fspt_stats(fspt_stats *stats);

#endif /* FSPT_H */
