#include "fspt.h"

#include <stdlib.h>
#include <assert.h>

static float volume(int n_features, const float *feature_limit)
{
    float vol = 0;
    for (int i = 0; i < 2*n_features; i++)
        vol *= feature_limit[i+1] - feature_limit[i];
    return vol;
}

fspt_t *make_fspt(int n_features, const float *feature_limit,
                  float *feature_importance, void (*criterion),
                  int min_samples_leaf, int max_depth, float gain_thresh)
{
    if (!feature_importance) {
        feature_importance = malloc(n_features * sizeof(float));
        float *ptr = feature_importance;
        while (*ptr < feature_importance + n_features) {
            *ptr = 1.;
            ptr++;
        }
    }
    fspt_t *fspt = calloc(1, sizeof(fspt));
    fspt->n_features = n_features;
    fspt->feature_limit = feature_limit;
    fspt->feature_importance = feature_importance;
    fspt->criterion = criterion;
    fspt->vol = volume(n_features, feature_limit);
    /* Not sure to initialize with a root node... */
    fspt->root = calloc(1,sizeof(fspt_node));
    fspt->root->type = LEAF;
    fspt->root->id = 0;
    fspt->root->n_features = n_features;
    fspt->root->feature_limit = feature_limit;
    return fspt;
}

/**
 * Gives the nodes containing each input X. The input parameter nodes
 * needs to be freed by the caller. But not the nodes themselves.
 * \param n The number of test samples in X.
 * \param fspt The feature space partitioning tree.
 * \param X Size (n * fspt->n_features), containing the inputs to test.
 * \param nodes Output parameter that will be filled by the n nodes.
 *              Needs to be freed by the caller.
 */ 
void fspt_decision_func(int n, const fspt_t *fspt, const float *X,
                        fspt_node **nodes)
{
    int n_features = fspt->n_features;
    for (int i = 0; i < n; i++) {
        const float *x = X + i * n_features;
        fspt_node *tmp_node = fspt->root;
        int not_found = 0;
        while (tmp_node->type != LEAF) {
            int split_feature = tmp_node->split_feature;
            if (x[split_feature] <= tmp_node->thresh_left) {
                tmp_node = tmp_node->left;
            } else if (x[split_feature] >= tmp_node->thresh_right) {
                tmp_node = tmp_node->right;
            } else {
                nodes[i] = NULL;
                continue;
            }
        }
        if (tmp_node->type == LEAF)
        {
            nodes[i] = tmp_node;
        }
    }
}

void fspt_predict(int n, const fspt_t *fspt, const float *X, float *Y)
{
    fspt_node **nodes = malloc(n * sizeof(fspt_node *));
    fspt_decision_func(n, fspt, X, nodes);
    for (int i = 0; i < n; i++) {
        if (nodes[i] == NULL) {
            Y[i] = 0.;
        } else {
            //TODO(Gab)
            Y[i] = score(nodes[i]);
        }
    }
    free(nodes);
}

void fspt_fit(int n, const float *X, float max_feature, float max_try,
              fspt_t *fspt)
{
    assert(fspt->max_depth >= 1);
    fspt->n_features = n;
    fspt_node *root = calloc(1, sizeof(fspt_node));
    root->id = 0;
    root->type = LEAF;
    root->n_features = n;
    root->feature_limit = fspt->feature_limit;
    root->n_samples = n;
    root->samples = X;
    fspt->root = root;
}
