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
    for (int i = 0; i < n_features; i+=2)
        vol *= feature_limit[i+1] - feature_limit[i];
    return vol;
}

float *get_feature_limit(const fspt_node *node) {
    float *feature_limit;
    if (node->parent) {
        int lim_index;
        if (node == node->parent->left) {
            lim_index = 2 * node->parent->split_feature + 1;
        } else {
            lim_index = 2 * node->parent->split_feature;
        }
        feature_limit = get_feature_limit(node->parent);
        feature_limit[lim_index] = node->parent->split_value;
    } else {
        feature_limit = malloc(2 * node->fspt->n_features * sizeof(float));
        copy_cpu(2 * node->fspt->n_features,
                (float *) node->fspt->feature_limit, 1,
                feature_limit, 1);
    }
    return feature_limit;
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
    while (X[split_index * n_features + index] <= s) {
        ++split_index;
        if (split_index == node->n_samples) break;
    }
    /* fill right node */
    right->type = LEAF;
    right->n_features = n_features;
    right->n_samples = node->n_samples - split_index;
    right->samples = X + split_index * n_features;
    right->n_empty = right->n_samples;
    right->depth = node->depth + 1;
    right->parent = node;
    right->fspt = fspt;
    right->score = fspt->score(fspt, right);
    /* fill left node */
    left->type = LEAF;
    left->n_features = n_features;
    left->n_samples = split_index;
    left->samples = X;
    left->n_empty = left->n_samples;
    left->depth = node->depth + 1;
    left->parent = node;
    left->fspt = fspt;
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

/**
 * Function to print binary tree in 2D
 * It does reverse inorder traversal
 * taken from : https://www.geeksforgeeks.org/print-binary-tree-2-dimensions/
 *
 * \param root Root of a subtree.
 * \param space indent to print this node
 */
static void print2DUtil(fspt_node *root, int space)
{
    const int COUNT = 10;
    if (root == NULL)
        return;
    space += COUNT;
    print2DUtil(root->right, space);
    fprintf(stderr, "\n");
    for (int i = COUNT; i < space; i++)
        fprintf(stderr, " ");
    fprintf(stderr, "%p\n", root);
    for (int i = COUNT; i < space; i++)
        fprintf(stderr, " ");
    if (root->type == INNER)
        fprintf(stderr, "%2d|%4.2f (%d)\n", root->split_feature, root->split_value, root->n_samples);
    else
        fprintf(stderr, "%4.3f(%d spl)\n", root->score, root->n_samples);
    print2DUtil(root->left, space);
}

void print_fspt(fspt_t *fspt)
{
    fprintf(stderr,
            "fspt %p: %d features, %d nodes, %d samples, %d depth\n",
            fspt, fspt->n_features, fspt->n_nodes, fspt->n_samples,
            fspt->depth);
    print2DUtil(fspt->root, 0);
}

fspt_t *make_fspt(int n_features, const float *feature_limit,
                  const float *feature_importance, criterion_func criterion,
                  score_func score)
{
    if (!feature_importance) {
        feature_importance = malloc(n_features * sizeof(float));
        float *ptr = (float *) feature_importance;
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
    return fspt;
}

void fspt_decision_func(int n, const fspt_t *fspt, const float *X,
                        fspt_node **nodes)
{
    int n_features = fspt->n_features;
    for (int i = 0; i < n; i++) {
        const float *x = X + i * n_features;
        fspt_node *tmp_node = fspt->root;
        if (!tmp_node) {
            nodes[i] = 0;
            continue;
        }
        while (tmp_node->type != LEAF) {
            int split_feature = tmp_node->split_feature;
            if (x[split_feature] < tmp_node->split_value) {
                tmp_node = tmp_node->left;
            } else if (x[split_feature] >= tmp_node->split_value) {
                tmp_node = tmp_node->right;
            }
        }
        if (tmp_node->type == LEAF) {
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
            debug_print("predicted score %f", nodes[i]->score);
            Y[i] = nodes[i]->score;
        }
    }
    free(nodes);
}

void fspt_fit(int n_samples, float *X, criterion_args *args, fspt_t *fspt)
{
    args->fspt = fspt;
    if (fspt->root)
        free_fspt_nodes(fspt->root);
    /* Builds the root */
    fspt_node *root = calloc(1, sizeof(fspt_node));
    root->type = LEAF;
    root->n_features = fspt->n_features;
    root->n_samples = n_samples;
    root->n_empty = (float)n_samples; //Arbitray initialize s.t. Density=0.5
    root->samples = X;
    root->depth = 1;
    root->fspt = fspt;
    /* Update fspt */
    fspt->n_nodes = 1;
    fspt->n_samples = n_samples;
    fspt->samples = X;
    fspt->root = root;
    fspt->depth = 1;
    if (!n_samples) return;

    list *fifo = make_list(); // fifo of the nodes to examine
    list_insert(fifo, (void *)root);
    while (fifo->size > 0) {
        fspt_node *current_node = (fspt_node *) list_pop(fifo);
        args->node = current_node;
        int *index = &args->best_index;
        float *s = &args->best_split;
        float *gain = &args->gain;
        /* fills the values of *args */
        fspt->criterion(args);
        // TO DELETE ONCE PROBLEM FIXED
        if (*gain > 0.5f) fprintf(stderr, "error : gain = %f\n", *gain);
        debug_assert((*gain <= 0.5f) && (0.f <= *gain));
        if (args->forbidden_split) {
            debug_print("forbidden split node %p", current_node);
            current_node->score = fspt->score(fspt, current_node);
        } else {
            debug_print("best_index=%d, best_split=%f, gain=%f",
                    *index, *s, *gain);
            fspt_node *left = calloc(1, sizeof(fspt_node));
            fspt_node *right = calloc(1, sizeof(fspt_node));
            fspt_split(fspt, current_node, *index, *s, left, right);
            list_insert_front(fifo, right);
            list_insert_front(fifo, left);
        }
    }
    free_list(fifo);
}

/**
 * Recursively saves from node in fp.
 *
 * \param fp A file pointer. Open and close file is caller's responsibility.
 * \param node The node from where we save.
 * \param succ Output parameter. Will contain 1 if successfully save,
 *             0 otherwise.
 */
static void pre_order_node_save(FILE *fp, fspt_node node, int *succ) {
    /* save node */
    node.samples = NULL;
    *succ &= fwrite(&node, sizeof(fspt_node), 1, fp);
    /* save children */
    if (node.left) pre_order_node_save(fp, *node.left, succ);
    if (node.right) pre_order_node_save(fp, *node.right, succ);
}

void fspt_save_file(FILE *fp, fspt_t fspt, int save_samples, int *succ) {
    /* save n_features */
    *succ &= fwrite(&fspt.n_features, sizeof(int), 1, fp);
    /* save feature_limit */
    size_t size = 2 * fspt.n_features;
    *succ &= (fwrite(fspt.feature_limit, sizeof(float), size, fp) == size);
    /* save feature_importance */
    size = fspt.n_features;
    *succ &=
        (fwrite(fspt.feature_importance, sizeof(float), size, fp) == size);
    /* save others */
    *succ &= fwrite(&fspt.n_nodes, sizeof(int), 1, fp);
    *succ &= fwrite(&fspt.n_samples, sizeof(int), 1, fp);
    *succ &= fwrite(&fspt.depth, sizeof(int), 1, fp);
    *succ &= fwrite(&fspt.count, sizeof(int), 1, fp);
    /* to know if the file contains samples */
    *succ &= fwrite(&save_samples, sizeof(int), 1, fp);
    /* save samples if requested */
    size = fspt.n_samples * fspt.n_features;
    if (save_samples && size) {
        *succ &= (fwrite(fspt.samples, sizeof(float), size, fp) == size);
    }
    if (fspt.root)
        pre_order_node_save(fp, *fspt.root, succ);
}

void fspt_save(const char *filename, fspt_t fspt, int save_samples, int *succ){
    *succ = 1;
    fprintf(stderr, "Saving fspt to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);
    fspt_save_file(fp, fspt, save_samples, succ);
    fclose(fp);
}

/**
 * Recursively loads from node in fp.
 *
 * \param fp A file pointer. Open and close file is caller's responsibility.
 * \param succ Output parameter. Will contain 1 if successfully load,
 *             0 otherwise.
 * \param n_samples The number of samples in samples.
 * \param samples A pointer to already existing and orderer samples or NULL.
 * \return Pointer to the newly created fspt_node.
 */
static fspt_node * pre_order_node_load(FILE *fp, int n_samples, float *samples,
        int *succ) {
    /* load node */
    fspt_node *node = malloc(sizeof(fspt_node));
    *succ &= fread(node, sizeof(fspt_node), 1, fp);
    if (!*succ) return NULL;
    /* point on samples */
    node->samples = samples;
    node->n_samples = n_samples;
    /* load children */
    float *samples_r = NULL;
    float *samples_l = NULL;
    int n_samples_r = 0;
    int n_samples_l = 0;
    if (node->left) {
        if (samples && n_samples) {
            int split_index = 0;
            while(samples[split_index * node->n_features + node->split_feature]
                    <= node->split_value) {
                ++split_index;
                if (split_index == n_samples) break;
            }
            samples_l = samples;
            samples_r = samples + split_index * node->n_features;
            n_samples_l = split_index;
            n_samples_r = n_samples - split_index;
        }
    }
    if (node->left)
        node->left = pre_order_node_load(fp, n_samples_l, samples_l, succ);
    if (node->right)
        node->right = pre_order_node_load(fp, n_samples_r, samples_r, succ);
    return node;
}

void fspt_load_file(FILE *fp, fspt_t *fspt, int load_samples, int *succ) {
    /* load n_features */
    int old_n_features = fspt->n_features;
    *succ &= fread(&fspt->n_features, sizeof(int), 1, fp);
    *succ &= (!old_n_features || (old_n_features == fspt->n_features));
    /* load feature_limit */
    size_t size = 2 * fspt->n_features;
    float *feature_limit = malloc(size * sizeof(float));
    *succ &=
        (fread(feature_limit, sizeof(float), size, fp) == size);
    fspt->feature_limit = feature_limit;
    /* load feature_importance */
    size = fspt->n_features;
    float *feature_importance = malloc(size * sizeof(float));
    *succ &= (fread(feature_importance, sizeof(float), size, fp) == size);
    fspt->feature_importance = feature_importance;
    /* load others */
    *succ &= fread(&fspt->n_nodes, sizeof(int), 1, fp);
    *succ &= fread(&fspt->n_samples, sizeof(int), 1, fp);
    *succ &= fread(&fspt->depth, sizeof(int), 1, fp);
    *succ &= fread(&fspt->count, sizeof(int), 1, fp);
    /* to know if file contains samples */
    fspt->samples = NULL;
    int contains_samples = 0;
    *succ &= fread(&contains_samples, sizeof(int), 1, fp);
    if (load_samples && !contains_samples)
        fprintf(stderr, "This file does not contain samples... continuing\n");
    if (contains_samples) {
        size = fspt->n_samples * fspt->n_features;
        if (load_samples && size) {
            float *samples = malloc(size * sizeof(float));
            *succ &= (fread(samples, sizeof(float), size, fp) == size);
            fspt->samples = samples;
        } else {
            fseek(fp, size *sizeof(float), SEEK_CUR);
        }
    }
    fspt->root = pre_order_node_load(fp, fspt->n_samples, fspt->samples, succ);
}

void fspt_load(const char *filename, fspt_t *fspt, int load_samples,
        int *succ) {
    fprintf(stderr, "Loading fspt from %s\n", filename);
    *succ = 1;
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    fspt_load_file(fp, fspt, load_samples, succ);
    fclose(fp);
}

void free_fspt_nodes(fspt_node *node) {
    if (!node) return;
    free_fspt_nodes(node->right);
    free_fspt_nodes(node->left);
    free(node);
}

void free_fspt(fspt_t *fspt) {
    free((float *) fspt->feature_limit);
    free((float *) fspt->feature_importance);
    free_fspt_nodes(fspt->root);
    free(fspt->samples);
    free(fspt);
}

