#ifndef FSPT_H
#define FSPT_H

typedef enum {LEAF, INNER} FSTP_NODE_TYPE;

struct fspt_node;

typedef struct fspt_node {
    FSTP_NODE_TYPE type;  // LEAF or INNER
    int id;               // id in the FSPT
    int n_features;
    float *feature_limit; // size 2*n_feature: feature_limit[2*i] = min feature(i)
                          // feature_limite[2*i+1] = max feature(i)
    float thresh_left;    // go to left child if feature[i] <= thresh_left
    float thresh_right;   // go to right child if feature[i] >= thresh_right

    struct fspt_node *right;   // right child
    struct fspt_node *left;    // left child

    int n_samples;
    float *samples;     // training samples

    int n_empty;        // number of empty points
    int depth;
    float vol;          // volume of the node (=prod length of each dimension)
    float density;      // density = n_samples/(n_samples + n_empty)
    int split_feature;  // splits on feature SPLIT_FEATURE

    int *potential_split_set; // ??

    int count;          // keeps the successive violation of gain threshold
} fspt_node;

typedef struct fspt_t {
    int n_features;       // number of features
    float *feature_limit;// size 2*n_feature: feature_limit[2*i] = min feature(i)
                         // feature_limite[2*i+1] = max feature(i)
    float *feature_importance; // feature_importance of size n_feature

    int n_nodes;          // number of nodes
    int *feature_split;  // feature_split[i] = split index for node i
    float *thresh_left;  // thresh_left[i] = split threshold for node i <=sL
    float *thresh_right; // thresh_right[i] = split threshold for node i>= sL
    fspt_node *child_left;  // child_letf[i] = left child of node [i]
    fspt_node *child_right; // child_right[i] = right child of node [i]
    fspt_node *root;

    void *criterion;     // spliting criterion
    float vol;           // volume of the tree
} fspt_t;

#endif /* FSPT_H */
