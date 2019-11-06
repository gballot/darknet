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

typedef enum {LEAF, INNER} FSTP_NODE_TYPE;
typedef enum {PRE_ORDER, IN_ORDER, POST_ORDER} FSPT_TRAVERSAL;


struct fspt_node;
struct fspt_t;
struct criterion_args;
typedef void (*criterion_func) (struct criterion_args *args);
typedef float (*score_func) (const struct fspt_node *node);

/**
 * Node of the FSPT.
 */
typedef struct fspt_node {
    FSTP_NODE_TYPE type;  // LEAF or INNER
    int n_features;
    struct fspt_t *fspt;   // the fspt that contains this node
    float n_empty;        // number of empty points (is a float)
    int n_samples;
    float *samples;     // training samples
    int split_feature;  // splits on feature SPLIT_FEATURE
    float split_value;   // go to right child if feature[i] > split_value
                         // to the left chil if feature[i] <= split_value.
    struct fspt_node *right;   // right child
    struct fspt_node *left;    // left child
    struct fspt_node *parent;  // the parent node
    int depth;
    float score;
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
    int n_nodes;         // number of nodes
    int n_samples;       // number of training samples
    float *samples;     // training samples
    fspt_node *root;
    criterion_func criterion; // spliting criterion
    score_func score;    // score_function
    int depth;
    int count;          // keeps the successive violation of gain threshold
    double volume;      // total volume of the fspt
    int min_samples;
    int max_depth;
} fspt_t;


typedef struct criterion_args {
    fspt_t *fspt;
    fspt_node *node;
    float max_tries_p;
    float max_features_p;
    float gini_gain_thresh;
    int max_depth;
    int min_samples;
    int best_index;
    float best_split;
    float gain;
    int forbidden_split;
    int end_of_fitting;
} criterion_args;

typedef struct fspt_infos {
    /* Inputs */
    fspt_t *fspt;                 // Fspt related to this infos.
    int n_thresh;                 // Number of thresholds.
    float *fspt_thresh;           // Score thresholds for some statistics.
    /* Volume statistics */
    double volume;                // Volume of the fspt.
    double *volume_above_thresh;  // Size n_thresh. Sum volume of leaves with
                                  // score above each thresholds.
    double *volume_above_thresh_p;  // Size n_thresh. Proportion of the sum of
                                    // the volume of the leaves with score
                                    // above each thresholds.
    /* Number of samples statistics */
    int n_samples;                // Number of samples of the tree.
    int min_samples_param;        // Parameter of the minimum number of
                                  // sample per nodes.
    int min_samples_leaves;       // Minimum number of samples per leaves.
    int max_samples_leaves;       // Maximum number of samples per leaves.
    int mean_samples_leaves;      // Mean number of samples per leaves.
    int median_samples_leaves;    // Median number of samples per leaves.
    int first_quartile_samples_leaves; // First Q number of samples per leaves.
    int third_quartile_samples_leaves; // Thid Q number of samples per leaves.
    int *n_samples_above_thresh;  // Number of samples in leaves with score
                                  // above each thresholds.
    int *n_samples_above_thresh_p; // Proportional number of samples in leaves
                                   // with score above each thresholds.
    /* Depth statistics */
    int max_depth;           // Max depth parameter of the tree.
    int depth;               // Depth of the tree.
    int mean_depth_leaves;   // Mean depth of the leaves.
    int min_depth_leaves;    // Min depth of the leaves.
    int median_depth_leaves; // Median depth of the leaves.
    int first_quartile_depth_leaves;  // First Q of depth of leaves.
    int third_quartile_depth_leaves;  // Third Q of depth of leaves.
    float balanced_index;    // Score between 0 and 1. 0 if line tree,
                             // 1 if balanced. (1 - (2*depth-1)/n_nodes).
    /* Node type statistics */
    int n_leaves;   // Number of leaves.
    int n_inner;    // Number of inner nodes.
    /* Split statistics */
    int *split_features_count;    // Size n_features. Value at index i
                                  // is the number of split on feature i.
    float *min_split_values;      // Min split value by features.
    float *max_split_values;      // Max split value by features.
    float *mean_split_values;     // Mean split value by features.
    float *median_split_values;   // Median split value by features.
    float *first_quartile_split_values; // Fist Q split value by features.
    float *third_quartile_split_values; // Third Q split value by features.
    /* Score statistics */
    float min_sore;      // Min score of leaves.
    float max_score;     // Max score of leaves.
    float mean_score;    // Mean score of leaves.
    float median_score;  // Median score of leaves.
    float first_quartile_score;  // Fist Q score of leaves.
    float third_quartile_score;  // Third Q score of leaves.
} fspt_infos;

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
extern void fspt_decision_func(int n, const fspt_t *fspt, const float *X,
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
extern void fspt_predict(int n, const fspt_t *fspt, const float *X, float *Y);

/**
 * Fits the feature space partitioning tree to the data X.
 *
 * \param n_samples The number of samples in X.
 * \param X the samples to fit.
 * \param args Pointer to the criterion args.
 * \param fspt The feature space partitioning tree.
 */
extern void fspt_fit(int n_samples, float *X, criterion_args *args,
        fspt_t *fspt);


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
 * \param succ Output parameter. True if succes, false otherwise.
 */
extern void fspt_load(const char *filename, fspt_t *fspt, int load_samples,
        int *succ);

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
 * \param succ Output parameter. True if succes, false otherwise.
 */
extern void fspt_load_file(FILE *fp, fspt_t *fspt, int load_samples,
        int *succ);

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

#endif /* FSPT_H */
