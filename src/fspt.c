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
    float vol = 1;
    for (int i = 0; i < 2*n_features; i++)
        vol *= feature_limit[i+1] - feature_limit[i];
    return vol;
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
    fspt_t *fspt = calloc(1, sizeof(fspt_t));
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
    if (!n_samples) return;
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
    fspt->samples = X;
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
        /* fills the values of *args */
        fspt->criterion(args);
        debug_print("best_index=%d, best_split=%f, gain=%f",*index,*s,*gain);
        if (*index == FAIL_TO_FIND) {
            //TODO
        } else {
            fspt_node *left = calloc(1, sizeof(fspt_node));
            fspt_node *right = calloc(1, sizeof(fspt_node));
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

static void post_order_node_save(FILE *fp, fspt_node node, int *succ) {
    if (node.left) post_order_node_save(fp, *node.left, succ);
    if (node.right) post_order_node_save(fp, *node.right, succ);
    /* save node */
    node.feature_limit = NULL;
    node.samples = NULL;
    *succ &= fwrite(&node, sizeof(fspt_node), 1, fp);
    /* save feature_limit */
    size_t lim_size = 2 * node.n_features;
    *succ &=
        (fwrite(node.feature_limit, sizeof(float), lim_size, fp) == lim_size);
}

void fspt_save(char *filename, fspt_t fspt, int *succ) {
    *succ = 1;
    fprintf(stderr, "Saving fspt to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);
    *succ &= fwrite(&fspt.n_nodes, sizeof(int), 1, fp);
    *succ &= fwrite(&fspt.n_samples, sizeof(int), 1, fp);
    *succ &= fwrite(&fspt.depth, sizeof(int), 1, fp);
    if (fspt.root)
        post_order_node_save(fp, *fspt.root, succ);
    fclose(fp);
}

static fspt_node * post_order_node_load(FILE *fp, int *succ) {
    /* load node */
    fspt_node *node = malloc(sizeof(fspt_node));
    *succ &= fread(node, sizeof(fspt_node), 1, fp);
    if (!*succ) return NULL;
    /* load feature_limit */
    size_t lim_size = 2 * node->n_features;
    float *feature_limit = malloc(lim_size * sizeof(float));
    *succ &=
        (fread(feature_limit, sizeof(float), lim_size, fp) != lim_size);
    /* load children */
    if (node->left) node->left = post_order_node_load(fp, succ);
    if (node->right) node->right = post_order_node_load(fp, succ);
    return node;
}

void fspt_load(char *filename, fspt_t *fspt, int *succ) {
    fprintf(stderr, "Loading fspt from %s\n", filename);
    *succ = 1;
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);
    *succ &= fread(&fspt->n_nodes, sizeof(int), 1, fp);
    *succ &= fread(&fspt->n_samples, sizeof(int), 1, fp);
    *succ &= fread(&fspt->depth, sizeof(float), 1, fp);
    fspt->root = post_order_node_load(fp, succ);
    fspt->vol = volume(fspt->n_features, fspt->feature_limit);
    fclose(fp);
}


