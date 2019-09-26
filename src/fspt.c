#include "fspt.h"

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "list.h"
#include "utils.h"



/**
 * Computes the volume of a feature space.
 * Volume = Prod_i(max feature[i] - min feature[i])
 *
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


static void compute_best_gain(size_t n_bins, const float *bins,
        const size_t *cdf, int n_samples, int n_empty,
        float current_score, float node_min, float node_max,
        int min_samples, float (*criterion) (void *),
        float *best_gain, size_t *best_gain_index) {
    //TODO: could add max_try policy.
    //Add an argument 0<max_try_p<1
    //and take a subsample of bins of size floor(max(1,n_bins*max_try_p))
    for (int j = 0; j < n_bins; ++j) {
        assert(n_empty == n_samples);
        float bin = bins[j];
        size_t n_left = cdf[j];
        size_t n_right = n_samples - cdf[j];
        gini_criterion_arg arg = {
            node_min,
            node_max,
            bin,
            n_left,
            n_right,
            n_empty,
            min_samples
        };
        float score = score((void *) &arg);
        float tmp_gain = current_score - score;
        if (tmp_gain > *best_gain) {
            *best_gain = tmp_gain;
            *best_gain_index = j;
        }
    }
}

/**
 * Finds the best_feature_index and the best_split that maximize the spliting
 * criterion.
 *
 * \param fspt The feature space partitioning tree.
 * \param node The node we will split.
 * \param max_try_p The percentage of sliting points among the potential split
 *                  value set that we want to try.
 * \param max_feature_p The percentage of random feature we want to try.
 * \param thresh The minimum gain we want to achieve.
 * \param best_feature_index Output parameter. Will be filled with the index
 *                           of the best feature to split on.
 * \param best_gain Output parameter. Will be filled with the best gain
 *                  achieved.
 * \param best_split Output parameter. Will be filled sith the best split
 *                   value.
 */
/*
static void best_spliter(const fspt_t *fspt, fspt_node *node,
                         float max_try_p, float max_feature_p, float thresh,
                         int *best_feature_index, float *best_gain,
                         float *best_split) {
    float *best_gains = malloc(fspt->n_features * sizeof(float));
    float *best_splits = malloc(fspt->n_features * sizeof(float));
    int *random_features = random_index_order(0, fspt->n_features - 1);
    float *X = node->samples;
    float current_score = 0.5; // Max of Gini index.
    //TODO don't go to n_features but floor(n_features*max_features_p)
    //int max_features = floor(fspt->n_features * max_features_p);
    for (int i = 0; i < fspt->n_features; ++i) {
        int feat = random_features[i];
        float node_min = node->feature_limit[2*feat];
        float node_max = node->feature_limit[2*feat + 1];
        float *bins;
        size_t *cdf;
        size_t n_bins;
        hist(node->n_samples, fspt->n_features, X + feat, node_min, &n_bins,
                cdf, bins);
        if (n_bins < 1) continue;
        size_t best_gain_index = 0;
        float best_gain = 0.;
        compute_best_gain(n_bins, bins, cdf, node->n_samples, node->n_empty,
                current_score, node_min, node_max, fspt->min_samples,
                fspt->criterion, &best_gain, &best_gain_index);
        float fspt_min = fspt->feature_limit[2*feat];
        float fspt_max = fspt->feature_limit[2*feat + 1];
        float relative_length = (node_max - node_min) / (fspt_max - fspt_min);
        best_gains[feat] = best_gain * fspt->feature_importance[feat]
            * relative_length;
        best_splits[feat] = bins[best_gain_index];
    }
    *best_feature_index = max_index(best_gains, fspt->n_features);
    *best_gain = best_gains[*best_feature_index];
    *best_split = best_splits[*best_feature_index];
    if (*best_gain < 0) {
        *best_feature_index = FAIL_TO_FIND;
        debug_print("fail to find any split point ad depth %d and n_samples %d",
                node->depth, node->n_samples);
    } else if (*best_gain < thresh) {
        debug_print("fail to find any split point ad depth %d and count %d",
                node->depth, node->count);
        node->count += 1;
        int v = 10 > fspt->n_samples / 500 ? v : fspt->n_samples / 500;
        if (node->count >= v) {
            *best_feature_index = FAIL_TO_FIND;
        }
    } else {
        node->count = 0;
    }

    free(random_features);
}
*/

/**
 * Helper funciton for the Quick Sort algorithm.
 * Puts smaller values before a choosen pivot and greater values after.
 *
 * \param index The index of the feature to apply QSort. 0 <= index < size.
 * \param n The number of vectors in the array.
 * \param size The number of feature of each vectors.
 * \param base Output paramter. Pointer to the array of size (n_size).
 * \return the index of the pivot in the output parameter base.
 */
static int partition(size_t index, size_t n, size_t size, float *base) {
    float *pivot = malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) {
        pivot[i] = base[(n/2) * n + i];
    }
    int i = -1;
    int j = n;
    while(1) {
        do { ++i; } while (base[i*size + index] < pivot[index]);
        do { --j; } while (base[j*size + index] > pivot[index]);
        if (i >= j) return j;
        /* Swap i and j. */
        for (size_t k = 0; k < size; ++k) {
            float swap = base[i*size + k];
            base[i*size + k] = base[j*size + k];
            base[j*size + k] = swap;
        }
    }
    free(pivot);
}

/**
 * Implementation of the Quick Sort algorithm on bidimensional arrays of
 * size (n*size) according to the feature index. Ascending order.
 *
 * \param index The index of the feature to apply QSort. 0 <= index < size.
 * \param n The number of vectors in the array.
 * \param size The number of feature of each vectors.
 * \param base Output paramter. Pointer to the array of size (n_size).
 */
static void qsort_float_on_index(size_t index, size_t n, size_t size,
                                 float *base) {
    if (n == 2) {
        if (base[index] > base[size + index]) {
            for (size_t i = 0; i < size; ++i) {
                float swap = base[i];
                base[i] = base[size + i];
                base[size + i] = swap;
            }
        }
    } else if (n > 2) {
        int p = partition(index, n, size, base);
        qsort_float_on_index(index, p, size, base);
        qsort_float_on_index(index, n - p, size, base + p);
    }
}

/**
 * Make a new split in the FSPT.
 * This function modifies the input/output parameter fspt, and the ouput
 * parameters right and left according to the split on feature `index`
 * on value `s`.
 *
 * \param fspt The FSPT build so far. His depth an n_nodes are modified.
 * \param node The current leaf node that will be splited. Is modified.
 * \param index The index of the feature that we split on.
 * \param s The value on features[index] that we split on.
 * \param right Output parameter. Filled according to the split.
 * \param left Output parameter. Filled according to the split.
 */
static void fspt_split(fspt_t *fspt, fspt_node *node, int index, float s,
                       fspt_node *right, fspt_node *left) {
    float *X = node->samples;
    int n_features = node->n_features;
    qsort_float_on_index(index, node->n_samples, n_features, X);
    int split_index = 0;
    while (X[split_index*n_features + index] < s)
        ++split_index;
    /* fill right node */
    right->type = LEAF;
    right->n_features = n_features;
    float * right_feature_limit = malloc(2*n_features);
    memcpy(right_feature_limit, node->feature_limit, 2*n_features);
    right_feature_limit[2*index] = s;
    right->feature_limit = right_feature_limit;
    right->n_samples = node->n_samples - split_index;
    right->samples = X + split_index;
    right->n_empty = node->n_empty * (node->feature_limit[2*index + 1] - s)
        / (node->feature_limit[2*index + 1] - node->feature_limit[2*index]);
    right->depth = node->depth + 1;
    right->vol = node->vol * (node->feature_limit[2*index + 1] - s)
        / (node->feature_limit[2*index + 1] - node->feature_limit[2*index]);
    right->density = right->n_samples / (right->n_samples + right->n_empty);
    right->score = fspt->score(fspt, right);
    /* fill left node */
    left->type = LEAF;
    left->n_features = n_features;
    float * left_feature_limit = malloc(2*n_features);
    memcpy(left_feature_limit, node->feature_limit, 2*n_features);
    left_feature_limit[2*index + 1] = s;
    left->feature_limit = left_feature_limit;
    left->n_samples = split_index;
    left->samples = X;
    left->n_empty = node->n_empty * (s - node->feature_limit[2*index])
        / (node->feature_limit[2*index + 1] - node->feature_limit[2*index]);
    left->depth = node->depth + 1;
    left->vol = node->vol * (s - node->feature_limit[2*index])
        / (node->feature_limit[2*index + 1] - node->feature_limit[2*index]);
    left->density = left->n_samples / (left->n_samples + left->n_empty);
    left->score = fspt->score(fspt, left);
    /* fill parent node */
    node->type = INNER;
    node->split_value = s;
    node->right = right;
    node->left = left;
    node->split_feature = index;
    /* update fspt */
    fspt->n_nodes += 2;
    if (right->depth > fspt->depth)
        fspt->depth = right->depth;
}

fspt_t *make_fspt(int n_features, const float *feature_limit,
                  float *feature_importance, criterion_func criterion,
                  score_func score, int min_samples, int max_depth)
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
    fspt->score = score;
    fspt->vol = volume(n_features, feature_limit);
    fspt->max_depth = max_depth;
    fspt->min_samples = min_samples;
    return fspt;
}

void fspt_decision_func(int n, const fspt_t *fspt, const float *X,
                        fspt_node **nodes)
{
    int n_features = fspt->n_features;
    for (int i = 0; i < n; i++) {
        const float *x = X + i * n_features;
        fspt_node *tmp_node = fspt->root;
        while (tmp_node->type != LEAF) {
            int split_feature = tmp_node->split_feature;
            if (x[split_feature] <= tmp_node->split_value) {
                tmp_node = tmp_node->left;
            } else if (x[split_feature] >= tmp_node->split_value) {
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
            Y[i] = nodes[i]->score;
        }
    }
    free(nodes);
}

void fspt_fit(int n_samples, float *X, criterion_args *args, fspt_t *fspt)
{
    assert(fspt->max_depth >= 1);
    args->fspt = fspt;
    /* Builds the root */
    fspt_node *root = calloc(1, sizeof(fspt_node));
    root->type = LEAF;
    root->n_features = fspt->n_features;
    root->feature_limit = fspt->feature_limit;
    root->n_samples = n_samples;
    root->n_empty = n_samples; // We arbitray initialize such that Density=0.5
    root->samples = X;
    root->depth = 1;
    root->vol = volume(root->n_features, root->feature_limit);
    /* Update fspt */
    fspt->n_nodes = 1;
    fspt->n_samples = n_samples;
    fspt->root = root;
    fspt->depth = 1;

    list *heap = make_list(); // Heap of the nodes to examine
    list_insert(heap, (void *)root);
    while (heap->size > 0) {
        fspt_node *current_node = (fspt_node *) list_pop(heap);
        int *index;
        float *s, *gain;
        args->node = current_node;
        index = &args->best_index;
        s = &args->best_split;
        gain = &args->gain;
        //best_spliter(fspt, current_node, 1., 1., 0.05, &index, &gain, &s);
        /* fills the values of *args */
        fspt->criterion(args);
        debug_print("best_index=%d, best_split=%f, gain=%f",*index,*s,*gain);
        if (*index == FAIL_TO_FIND) {
            //TODO
        } else {
            fspt_node *left, *right;
            fspt_split(fspt, current_node, *index, *s, left, right);
            /* Should I examine right ? */
            if (right->depth > fspt->max_depth
                    || right->n_samples < fspt->min_samples) {
                right->score = fspt->score(fspt, right);
            } else {
                list_insert(heap, right);
            }
            /* Should I examine left ? */
            if (left->depth > fspt->max_depth
                    || left->n_samples < fspt->min_samples) {
                left->score = fspt->score(fspt, left);
            } else {
                list_insert(heap, left);
            }
        }
    }
}
