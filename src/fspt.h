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

typedef enum {LEAF, INNER} FSTP_NODE_TYPE;

typedef struct criterion_args {
    fspt_t *fspt;
    fspt_node *node;
    int feature_index;
    float spliting_point;
    float max_try_p;
    float max_feature_p;
    float thresh;
    int *best_index;
    float *best_split;
    float *gain;
} criterion_args;

struct fspt_node;

/**
 * Node of the FSPT.
 */
typedef struct fspt_node {
    FSTP_NODE_TYPE type;  // LEAF or INNER
    int id;               // id in the FSPT
    int n_features;
    const float *feature_limit; // size 2*n_feature:
                          // feature_limit[2*i] = min feature(i)
                          // feature_limite[2*i+1] = max feature(i)
    float thresh_left;    // go to left child if feature[i] < thresh_left
    float thresh_right;   // go to right child if feature[i] >= thresh_right
    struct fspt_node *right;   // right child
    struct fspt_node *left;    // left child
    int n_samples;
    float *samples;     // training samples
    int n_empty;        // number of empty points
    int depth;
    float vol;          // volume of the node (=prod length of each dimension)
    float density;      // density = n_samples/(n_samples + n_empty)
    float score;
    int split_feature;  // splits on feature SPLIT_FEATURE
    int *potential_split_set; // ??
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
    int *feature_split;  // feature_split[i] = split index for node i
    float *thresh_left;  // thresh_left[i] = split threshold for node i <=sL
    float *thresh_right; // thresh_right[i] = split threshold for node i>= sL
    fspt_node *child_left;  // child_letf[i] = left child of node [i]
    fspt_node *child_right; // child_right[i] = right child of node [i]
    fspt_node *root;
    float (*criterion) (fspt_t *fspt, fspt_node *node, int feature_index,
            float s); // spliting criterion
    float (*score) (fspt_t *fspt, fspt_node *node);     // score_function
    float vol;           // volume of the tree
    int max_depth;
    int min_samples;
} fspt_t;


/**
 * Builds an empty feature space partitioning tree.
 * 
 * \param n_features The number of features.
 * \param feature_limit values at index i and i+1 are respectively
 *                      the min and max of feature i.
 * \param feature_importance The static feature importance. Initialized at 1.
 *                           if NULL.
 * \param criterion The evaluation function.
 * \param min_samples_leaf Lower bound of samples per leaf.
 * \param max_depth Upper bound of the depth of the tree.
 * \param gain_thresh ???
 * \return A pointer to the fspt_tree. Must be freed by the caller with
 *         a call to fspt_free(fspt_tree fspt).
 */
extern fspt_t *make_fspt(
        int n_features,
        const float *feature_limit,
        float *feature_importance,
        void (*criterion),
        int min_samples_leaf,
        int max_depth,
        float gain_thresh);

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
 * \param max_feature The maximum number feature to be visited (TODO).
 * \param max_try Maximum number of split ???.
 */
extern void fspt_fit(int n_samples, float *X,
              float max_feature, float max_try, fspt_t *fspt);

#endif /* FSPT_H */
