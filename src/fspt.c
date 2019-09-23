#include "fspt.h"

#include <stdlib.h>

fspt_t *make_fspt(int n_features, float *feature_limit,
        float *feature_importance, void * criterion, int min_samples_leaf,
        int max_depth, float gain_thresh)
{
    if(!feature_importance) {
        feature_importance = malloc(n_features * sizeof(float));
        float *ptr = feature_importance;
        while(ptr < feature_importance + n_features)
        {
            *ptr = 1.;
            ptr++;
        }
    }
    fspt_t *fspt = calloc(1, sizeof(fspt));
    fspt->n_features = n_features;
    fspt->feature_limit = feature_limit;
    fspt->feature_importance = feature_importance;
    fspt->criterion = criterion;
    return fspt;
}

void fspt_decision_func(int n, fspt_t *fspt, float *X, fspt_node **nodes) {
    int n_features = fspt->n_features;
    for(int i = 0; i < n; i++) {
        float *x = X + i * n_features;
        fspt_node *tmp_node = fspt->root;
        while (tmp_node->type != LEAF)
        {
            int split_feature = tmp_node->split_feature;
            if (x[split_feature] <= tmp_node->thresh_left)
                tmp_node = tmp_node->left;
            else if (x[split_feature] >= tmp_node->thresh_right)
                tmp_node = tmp_node->right;
            else 
                nodes[i] = NULL;
        }
        if (tmp_node->type == LEAF)
        {
            nodes[i] = tmp_node;
        }
    }
}

void fspt_predict(int n, fspt_t *fspt, float *X, float *Y) {
    
}
