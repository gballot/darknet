#include "fspt.h"

#include <assert.h>
#include <stdlib.h>

#include "list.h"

/**
 * Computes the volume of a fspt_node
 * Volume = Prod_i(max feature[i] - min feature[i])
 * \param n_features The number of features.
 * \param feature_limit values at index i and i+1 are respectively
 *                      the min and max of feature i.
 */
static float volume(int n_features, const float *feature_limit)
{
    float vol = 0;
    for (int i = 0; i < 2*n_features; i++)
        vol *= feature_limit[i+1] - feature_limit[i];
    return vol;
}

static float fspt_score(const fspt_t *fspt, fspt_node *node) {
    //TODO
    return 0.5;
}

static void best_spliter(const fspt_t *fspt, int *index, float *s,
                         float *gain) {
    //TODO
}

static void fspt_split(const fspt_t *fspt, fspt_node *node, int index, float s,
                       fspt_node *right, fspt_node *left) {
    //TODO
}

fspt_t *make_fspt(int n_features, const float *feature_limit,
                  float *feature_importance, void (*criterion),
                  int min_samples_leaf, int max_depth, float gain_thresh)
{
    if (!feature_importance) {
        feature_importance = malloc(n_features * sizeof(float));
        float *ptr = feature_importance;
        while (ptr < feature_importance + n_features) {
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
    return fspt;
}

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

void fspt_fit(int n_samples, const float *X,
              float max_feature, float max_try, fspt_t *fspt)
{
    assert(fspt->max_depth >= 1);
    /* Builds the root */
    fspt_node *root = calloc(1, sizeof(fspt_node));
    root->id = 0;
    root->type = LEAF;
    root->n_features = fspt->n_features;
    root->feature_limit = fspt->feature_limit;
    root->n_samples = n_samples;
    root->n_empty = n_samples; // We arbitray initialize such that Density=0.5
    root->samples = X;
    root->depth = 1;
    root->vol = volume(fspt->n_features, fspt->feature_limit);
    fspt->root = root;

    list *heap = make_list(); // Heap of the nodes to examine
    list_insert(heap, (void *)root);
    while (heap->size > 0) {
        fspt_node *current_node = (fspt_node *) list_pop(heap);
        int index;
        float s, gain;
        best_spliter(fspt, &index, &s, &gain);
        if (index == FAIL_TO_FIND) {
            //TODO
        } else {
            fspt_node *left, *right;
            fspt_split(fspt, current_node, index, s, left, right);
            /* Should I examine right ? */
            if (right->depth > fspt->max_depth
                    || right->n_samples < fspt->min_sample) {
                right->score = fspt_score(right);
            } else {
                list_insert(heap, right);
            }
            /* Should I examine left ? */
            if (left->depth > fspt->max_depth
                    || left->n_samples < fspt->min_sample) {
                left->score = fspt_score(left);
            } else {
                list_insert(heap, left);
            }
        }
    }
}
