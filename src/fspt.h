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

#define FAIL_TO_FIND -1

typedef enum {LEAF, INNER} FSTP_NODE_TYPE;


struct fspt_node;
struct fspt_t;
struct criterion_args;
typedef void (*criterion_func) (struct criterion_args *args);
typedef float (*score_func) (const struct fspt_t *fspt,
        const struct fspt_node *node);

/**
 * Node of the FSPT.
 */
typedef struct fspt_node {
    FSTP_NODE_TYPE type;  // LEAF or INNER
    int n_features;
    const float *feature_limit; // size 2*n_feature:
                          // feature_limit[2*i] = min feature(i) include
                          // feature_limite[2*i+1] = max feature(i) exclude
    float n_empty;        // number of empty points (is a float)
    int n_samples;
    float *samples;     // training samples
    int split_feature;  // splits on feature SPLIT_FEATURE
    float split_value;   // go to right child if feature[i] >= split_value
                         // to the left chil otherwise.
    struct fspt_node *right;   // right child
    struct fspt_node *left;    // left child
    int depth;
    float vol;          // volume of the node (=prod length of each dimension)
    float density;      // density = n_samples/(n_samples + n_empty)
    float score;
    int count;          // keeps the successive violation of gain threshold
} fspt_node;

/**
 * Feature Space Partitioning Tree.
 */
typedef struct fspt_t {
    int n_features;      // number of features
    const float *feature_limit;// size 2*n_feature:
                         //feature_limit[2*i] = min feature(i)
                         // feature_limite[2*i+1] = max feature(i)
    float *feature_importance; // feature_importance of size n_feature
    int n_nodes;         // number of nodes
    int n_samples;       // number of training samples
    float *samples;     // training samples
    fspt_node *root;
    criterion_func criterion; // spliting criterion
    score_func score;    // score_function
    float vol;           // volume of the tree
    int depth;
    int max_depth;
    int min_samples;
} fspt_t;


typedef struct criterion_args {
    fspt_t *fspt;
    fspt_node *node;
    float max_try_p;
    float max_feature_p;
    float thresh;
    int best_index;
    float best_split;
    float gain;
} criterion_args;

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
 * \param min_samples Lower bound of samples per leaf.
 * \param max_depth Upper bound of the depth of the tree.
 * \return A pointer to the fspt_tree. Must be freed by the caller with
 *         a call to fspt_free(fspt_tree fspt).
 */
extern fspt_t *make_fspt(
        int n_features,
        const float *feature_limit,
        float *feature_importance,
        criterion_func criterion,
        score_func score,
        int min_samples,
        int max_depth);

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
 * \param succ Output parameter. True if succes, false otherwise.
 */
extern void fspt_save(char *filename, fspt_t fspt, int *succ);


/**
 * Load the fspt from a file.
 * You must have created the fspt with @see make_fspt(), because
 * some fields of the fspt are assumed to be already filled.
 * This function fills n_nodes, n_samples, depth, vol and root.
 *
 * \param filename The path of the file to save.
 * \param fspt The feature space partitioning tree parsed from config file.
 * \param succ Output parameter. True if succes, false otherwise.
 */
extern void fspt_load(char *filename, fspt_t *fspt, int *succ);

#endif /* FSPT_H */
