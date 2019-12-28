#include "fspt.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "distance_to_boundary.h"
#include "fspt_criterion.h"
#include "fspt_score.h"
#include "list.h"
#include "uniformity.h"
#include "utils.h"

#define N_THRESH_STATS_FSPT 11
#define FLT_FORMAT "%12g"
#define LEFTFLTFOR "%-12g"
#define STR_FORMAT "%s"
#define INT_FORMAT "%12d"
#define LINTFORMAT "%12ld"
#define LEFTINTFOR "%-12d"
#define NODE_VERSION 3

/**
 * Computes the volume of a feature space.
 * Volume = Prod_i(max feature[i] - min feature[i])
 *
 * \param n_features The number of features.
 * \param feature_limit values at index i and i+1 are respectively
 *                      the min and max of feature i.
 */
static double volume(int n_features, const float *feature_limit) {
    double vol = 1;
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
 * Recursively inserts nodes into a list following the traversal.
 *
 * \param nodes The list to insert nodes.
 * \param node The current node.
 * \param traversal The mode of traversal @see FSPT_TRAVERSAL.
 */
static void add_nodes_to_list(list *nodes, fspt_node *node,
        FSPT_TRAVERSAL traversal, FSPT_NODE_TYPE type) {
    if (!node) return;
    if (traversal == PRE_ORDER) {
        if (type == ALL_NODES || type == node->type)
            list_insert(nodes, node);
    }
    add_nodes_to_list(nodes, node->left, traversal, type);
    if (traversal == IN_ORDER) {
        if (type == ALL_NODES || type == node->type)
            list_insert(nodes, node);
    }
    add_nodes_to_list(nodes, node->right, traversal, type);
    if (traversal == POST_ORDER) {
        if (type == ALL_NODES || type == node->type)
            list_insert(nodes, node);
    }
}

/**
 * Creates a list with the nodes of a fspt. The traversal mode can be
 * customized. The caller must free the list.
 *
 * \param fspt The fspt/
 * \param traversal The mode of traversal @see FSPT_TRAVERSAL.
 * \return The list of all the nodes.
 */
list *fspt_nodes_to_list(fspt_t *fspt, FSPT_TRAVERSAL traversal) {
    list *nodes = make_list();
    add_nodes_to_list(nodes, fspt->root, traversal, ALL_NODES);
    return nodes;
}

/**
 * Creates a list with the leaves of a fspt. The traversal mode can be
 * customized. The caller must free the list.
 *
 * \param fspt The fspt/
 * \param traversal The mode of traversal @see FSPT_TRAVERSAL.
 * \return The list of all the nodes.
 */
list *fspt_leaves_to_list(fspt_t *fspt, FSPT_TRAVERSAL traversal) {
    list *nodes = make_list();
    add_nodes_to_list(nodes, fspt->root, traversal, LEAF);
    return nodes;
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
    size_t split_index = 0;
    float *feature_limit = get_feature_limit(node);
    double d_feat = feature_limit[2*index + 1]
        - feature_limit[2*index];
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
    right->volume = node->volume * (feature_limit[2*index+1] - s) / d_feat;
    /* fill left node */
    left->type = LEAF;
    left->n_features = n_features;
    left->n_samples = split_index;
    left->samples = X;
    left->n_empty = left->n_samples;
    left->depth = node->depth + 1;
    left->parent = node;
    left->fspt = fspt;
    left->volume = node->volume * (s - feature_limit[2 * index]) / d_feat;
    /* fill parent node */
    node->type = INNER;
    node->split_value = s;
    node->right = right;
    node->left = left;
    node->split_feature = index;
    node->cause = SPLIT;
    /* update fspt */
    fspt->n_nodes += 2;
    if (right->depth > fspt->depth)
        fspt->depth = right->depth;
    free(feature_limit);
}

/**
 * Function to print binary tree in 2D
 * It does reverse inorder traversal
 * taken from : https://www.geeksforgeeks.org/print-binary-tree-2-dimensions/
 *
 * \param root Root of a subtree.
 * \param space indent to print this node
 */
static void print2DUtil(fspt_node *root, int space) {
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
        fprintf(stderr, "%2d|%4.2f (%ld)\n", root->split_feature, root->split_value, root->n_samples);
    else
        fprintf(stderr, "%4.3f(%ld spl)\n", root->score, root->n_samples);
    print2DUtil(root->left, space);
}

void print_fspt(fspt_t *fspt) {
    fprintf(stderr,
            "fspt %p: %d features, %ld nodes, %ld samples, %d depth\n",
            fspt, fspt->n_features, fspt->n_nodes, fspt->n_samples,
            fspt->depth);
    print2DUtil(fspt->root, 0);
}

fspt_t *make_fspt(int n_features, const float *feature_limit,
                  const float *feature_importance, criterion_func criterion,
                  score_func score) {
    if (!feature_importance) {
        feature_importance = malloc(n_features * sizeof(float));
        float *ptr = (float *) feature_importance;
        while (ptr < feature_importance + n_features) *ptr++ = 1.;
    }
    fspt_t *fspt = calloc(1, sizeof(fspt_t));
    fspt->n_features = n_features;
    fspt->feature_limit = feature_limit;
    fspt->feature_importance = feature_importance;
    fspt->criterion = criterion;
    fspt->score = score;
    fspt->volume = volume(n_features, feature_limit);
    return fspt;
}

/**
 * Helper function for qsort on volume of the nodes.
 *
 * \param n1 The first pointer to a node.
 * \param n2 The second pointer to a node.
 * \return Negative if n1 < n2, positive if n1 > n2, 0 if n1 == n2
 *         according to the volume of the nodes.
 */
static int cmp_volume_nodes(const void *n1, const void *n2) {
    fspt_node *node1 = *(fspt_node **) n1;
    fspt_node *node2 = *(fspt_node **) n2;
    if (node1->volume < node2->volume)
        return -1;
    else
        return (node1->volume > node2->volume);
}

/**
 * Helper function for median, first and third quartiles.
 *
 * \param n The pointer to node pointer.
 * \return the number of samples of the node.
 */
static double acc_volume(const void *n) {
    fspt_node *node = *(fspt_node **) n;
    return node->volume;
}

/**
 * Helper function for qsort on number of samples of the nodes.
 *
 * \param n1 The first pointer to a node pointer.
 * \param n2 The second pointer to a node pointer.
 * \return Negative if n1 < n2, positive if n1 > n2, 0 if n1 == n2
 *         according to the number of samples of the nodes.
 */
static int cmp_n_samples_nodes(const void *n1, const void *n2) {
    fspt_node *node1 = *(fspt_node **) n1;
    fspt_node *node2 = *(fspt_node **) n2;
    return node1->n_samples - node2->n_samples;
}

/**
 * Helper function for median, first and third quartiles.
 *
 * \param n The pointer to node pointer.
 * \return the number of samples of the node.
 */
static double acc_n_samples(const void *n) {
    fspt_node *node = *(fspt_node **) n;
    return (double) node->n_samples;
}

/**
 * Helper function for qsort on depth of the nodes.
 *
 * \param n1 The first pointer to a node pointer.
 * \param n2 The second pointer to a node pointer.
 * \return Negative if n1 < n2, positive if n1 > n2, 0 if n1 == n2
 *         according to the depth of the nodes.
 */
static int cmp_depth_nodes(const void *n1, const void *n2) {
    fspt_node *node1 = *(fspt_node **) n1;
    fspt_node *node2 = *(fspt_node **) n2;
    return node1->depth - node2->depth;
}

/**
 * Helper function for median, first and third quartiles.
 *
 * \param n The pointer to node pointer.
 * \return the number of samples of the node.
 */
static double acc_depth(const void *n) {
    fspt_node *node = *(fspt_node **) n;
    return (double) node->depth;
}

/**
 * Helper function for qsort on split value of the nodes.
 *
 * \param n1 The first pointer to a node pointer.
 * \param n2 The second pointer to a node pointer.
 * \return Negative if n1 < n2, positive if n1 > n2, 0 if n1 == n2
 *         according to the split value of the nodes.
 */
static int cmp_split_value_nodes(const void *n1, const void *n2) {
    fspt_node *node1 = *(fspt_node **) n1;
    fspt_node *node2 = *(fspt_node **) n2;
    if (node1->split_value < node2->split_value)
        return -1;
    else
        return (node1->split_value > node2->split_value);
}

/**
 * Helper function for median, first and third quartiles.
 *
 * \param n The pointer to node pointer.
 * \return the number of samples of the node.
 */
static double acc_split_value(const void *n) {
    fspt_node *node = *(fspt_node **) n;
    return (double) node->split_value;
}

/**
 * Helper function for qsort on score of the nodes.
 *
 * \param n1 The first pointer to a node pointer.
 * \param n2 The second pointer to a node pointer.
 * \return Negative if n1 < n2, positive if n1 > n2, 0 if n1 == n2
 *         according to the score of the nodes.
 */
static int cmp_score_nodes(const void *n1, const void *n2) {
    fspt_node *node1 = *(fspt_node **) n1;
    fspt_node *node2 = *(fspt_node **) n2;
    if (node1->score < node2->score)
        return -1;
    else
        return (node1->score > node2->score);
}

/**
 * Helper function for median, first and third quartiles.
 *
 * \param n The pointer to node pointer.
 * \return the number of samples of the node.
 */
static double acc_score(const void *n) {
    fspt_node *node = *(fspt_node **) n;
    return node->score;
}

/**
 * Helper function for qsort on score of the nodes.
 *
 * \param n1 The first pointer to a node pointer.
 * \param n2 The second pointer to a node pointer.
 * \return Negative if n1 < n2, positive if n1 > n2, 0 if n1 == n2
 *         according to the score of the nodes.
 */
static int cmp_score_vol_n(const void *n1, const void *n2) {
    score_vol_n *node1 = (score_vol_n *) n1;
    score_vol_n *node2 = (score_vol_n *) n2;
    if (node1->score > node2->score)
        return -1;
    else
        return (node1->score < node2->score);
}

fspt_stats *get_fspt_stats(fspt_t *fspt, int n_thresh, double *fspt_thresh,
        int do_uniformity_test) {
    if (!fspt) return NULL;
    if (!fspt->root) return NULL;
    /** Default values for thresh if NULL **/
    fspt_stats *stats = calloc(1, sizeof(fspt_stats));
    stats->fspt = fspt;
    if (n_thresh) {
        stats->n_thresh = n_thresh;
        if (fspt_thresh) {
            stats->fspt_thresh = fspt_thresh;
        } else {
            stats->fspt_thresh = malloc(n_thresh * sizeof(double));
            for (int i = 0; i < n_thresh; ++i) {
                stats->fspt_thresh[i] =
                    ((double) i) / ((double) (n_thresh - 1));
            }
        }
    } else {
        n_thresh = N_THRESH_STATS_FSPT;
        stats->n_thresh = n_thresh;
        stats->fspt_thresh = malloc(n_thresh * sizeof(double));
        for (int i = 0; i < n_thresh; ++i) {
            stats->fspt_thresh[i] = ((double) i) / ((double) (n_thresh - 1));
        }
    }
    fspt_thresh = stats->fspt_thresh;
    int n_nodes = fspt->n_nodes;
    int n_features = fspt->n_features;
    int n_samples = fspt->n_samples;

    /** Allocate arrays **/
    /* Volume */
    stats->volume_above_thresh = calloc(n_thresh, sizeof(double));
    stats->volume_above_thresh_p = calloc(n_thresh, sizeof(double));
    /* Samples */
    stats->n_samples_above_thresh = calloc(n_thresh, sizeof(size_t));
    stats->n_samples_above_thresh_p = calloc(n_thresh, sizeof(size_t));
    /* Depth */
    if (!fspt->depth) {
        fprintf(stderr, "fspt without depth");
        return stats;
    }
    stats->n_nodes_by_depth = calloc(fspt->depth, sizeof(size_t));
    stats->n_nodes_by_depth_p = calloc(fspt->depth, sizeof(double));
    /* Nodes */
    stats->n_leaves_above_thresh = calloc(n_thresh, sizeof(size_t));
    stats->n_leaves_above_thresh_p = calloc(n_thresh, sizeof(double));
    /* Splits */
    if (!fspt->n_features) {
        fprintf(stderr, "fspt without n_features");
        return stats;
    }
    stats->split_features_count = calloc(n_features, sizeof(size_t));
    stats->split_features_count_p = calloc(n_features, sizeof(double));
    stats->min_split_values = calloc(n_features, sizeof(double));
    stats->max_split_values = calloc(n_features, sizeof(double));
    stats->mean_split_values = calloc(n_features, sizeof(double));
    stats->median_split_values = calloc(n_features, sizeof(double));
    stats->first_quartile_split_values = calloc(n_features, sizeof(double));
    stats->third_quartile_split_values = calloc(n_features, sizeof(double));

    /** Nodes lists and arrays **/
    list *nodes = fspt_nodes_to_list(fspt, PRE_ORDER);
    fspt_node **nodes_array = (fspt_node **) list_to_array(nodes);
    list *leaves = make_list();
    list *inner_nodes = make_list();
    list **inner_nodes_by_split_feat = calloc(n_features, sizeof(list *));
    for (int i = 0; i < n_features; ++i) {
        inner_nodes_by_split_feat[i] = make_list();
    }
    for (int i = 0; i < n_nodes; ++i) {
        fspt_node *node = nodes_array[i];
        if (node->type == LEAF) {
            list_insert_front(leaves, node);
        } else if (node->type == INNER) {
            list_insert_front(inner_nodes, node);
            list_insert_front(inner_nodes_by_split_feat[node->split_feature],
                    node);
        } else {
            error("unkown node type");
        }
    }
    if (!leaves->size) {
        fprintf(stderr, "fspt without leaves");
        goto free_no_leaves;
    }
    fspt_node **leaves_array = (fspt_node **) list_to_array(leaves);
    fspt_node **inner_nodes_array = (fspt_node **) list_to_array(inner_nodes);
    fspt_node ***inner_nodes_by_split_feat_arrays =
        calloc(n_features, sizeof(fspt_node **));
    for (int i = 0; i < n_features; ++i) {
        inner_nodes_by_split_feat_arrays[i] =
            (fspt_node **) list_to_array(inner_nodes_by_split_feat[i]);
    }
    // allocate last array
    stats->score_vol_n_array = calloc(leaves->size, sizeof(score_vol_n));

    /** Means, nodes by depth, and thresholds statistics */
    for (int i = 0; i < leaves->size; ++i) {
        fspt_node *node = leaves_array[i];
        stats->leaves_volume += node->volume;
        stats->mean_samples_leaves += node->n_samples;
        stats->mean_depth_leaves += node->depth;
        stats->mean_score += node->score;
        double unf_score = 0;
        if (do_uniformity_test) {
            /*
            if (node->n_samples > (size_t) n_features) {
                struct unf_options options = {0};
                // TODO: put the right options in order to keep the feature limits.
                unf_score = unf_test_float(&options, node->samples,
                        node->n_samples, node->n_features);
            } else {
                unf_score = 1.;
            }
            */
            unf_score = dist_to_bound_test(n_features, node->n_samples,
                    node->samples, get_feature_limit(node));
        }
        stats->score_vol_n_array[i] =
            (score_vol_n) {
                node->score,
                node->volume / fspt->volume,
                node->n_samples,
                node->cause,
                unf_score
            };
        /* Thresholds */
        for (int j = 0; j < n_thresh; ++j) {
            double thresh = fspt_thresh[j];
            if (node->score >= thresh) {
                stats->volume_above_thresh[j] += node->volume;
                stats->n_samples_above_thresh[j] += node->n_samples;
                stats->n_leaves_above_thresh[j] += 1;
            }
        }
    }
    stats->mean_volume = stats->leaves_volume / leaves->size;
    stats->leaves_volume_p = stats->leaves_volume / fspt->volume;
    stats->mean_samples_leaves /= leaves->size;
    stats->mean_depth_leaves /= leaves->size;
    stats->mean_score /= leaves->size;
    stats->mean_volume_p = stats->mean_volume / fspt->volume;
    stats->mean_samples_leaves_p =
        n_samples ? stats->mean_samples_leaves / n_samples : 0.f;
    stats->mean_depth_leaves_p = stats->mean_depth_leaves / fspt->depth;
    qsort(stats->score_vol_n_array, leaves->size, sizeof(score_vol_n),
            cmp_score_vol_n);
    for (int j = 0; j < n_thresh; ++j) {
        stats->volume_above_thresh_p[j] =
            stats->volume_above_thresh[j] / fspt->volume;
        stats->n_samples_above_thresh_p[j] =
            ((double) stats->n_samples_above_thresh[j]) / n_samples;
        stats->n_leaves_above_thresh_p[j] =
            ((double) stats->n_leaves_above_thresh[j]) / leaves->size;
    }
    for (int i = 0; i < nodes->size; ++i) {
        fspt_node *node = nodes_array[i];
        stats->n_nodes_by_depth[node->depth - 1] += 1;
    }
    for (int i = 0; i < fspt->depth; ++i) {
        stats->n_nodes_by_depth_p[i] = 
            ((double) stats->n_nodes_by_depth[i]) / pow(2, i);
    }
    for (int feat = 0; feat < n_features; ++feat) {
        if (!inner_nodes_by_split_feat[feat]->size) continue;
        for (int i = 0; i < inner_nodes_by_split_feat[feat]->size; ++i) {
            fspt_node *node = inner_nodes_by_split_feat_arrays[feat][i];
            stats->mean_split_values[feat] += node->split_value;
        }
        stats->mean_split_values[feat] /= inner_nodes_by_split_feat[feat]->size;
    }


    /** Volume statistics **/
    qsort(leaves_array, leaves->size, sizeof(fspt_node *), cmp_volume_nodes);
    if (!fspt->volume) {
        fprintf(stderr, "fspt without volume");
        goto free_arrays;
    }
    stats->volume = fspt->volume;
    stats->min_volume = leaves_array[0]->volume;
    stats->min_volume_parameter =
        fspt->c_args ? fspt->c_args->min_volume_p : 0.;
    stats->max_volume = leaves_array[leaves->size - 1]->volume;
    stats->median_volume =
        median((const void *)leaves_array, leaves->size, sizeof(fspt_node *),
                acc_volume);
    stats->first_quartile_volume =
        first_quartile((const void *)leaves_array, leaves->size,
                sizeof(fspt_node *), acc_volume);
    stats->third_quartile_volume =
        third_quartile((const void *)leaves_array, leaves->size,
                sizeof(fspt_node *), acc_volume);
    stats->min_volume_p = stats->min_volume / fspt->volume;
    stats->max_volume_p = stats->max_volume / fspt->volume;
    stats->median_volume_p = stats->median_volume / fspt->volume;
    stats->first_quartile_volume_p =
        stats->first_quartile_volume / fspt->volume;
    stats->third_quartile_volume_p =
        stats->third_quartile_volume / fspt->volume;

    /** Number of samples statistics **/
    qsort(leaves_array, leaves->size, sizeof(fspt_node*), cmp_n_samples_nodes);
    stats->n_samples = fspt->n_samples;
    stats->min_samples_param = fspt->c_args ? fspt->c_args->min_samples : 0;
    stats->min_samples_leaves = leaves_array[0]->n_samples;
    stats->max_samples_leaves = leaves_array[leaves->size - 1]->n_samples;
    stats->median_samples_leaves =
        median((const void *)leaves_array, leaves->size, sizeof(fspt_node *),
                acc_n_samples);
    stats->first_quartile_samples_leaves =
        first_quartile((const void *)leaves_array, leaves->size,
                sizeof(fspt_node *), acc_n_samples);
    stats->third_quartile_samples_leaves =
        third_quartile((const void *)leaves_array, leaves->size,
                sizeof(fspt_node *), acc_n_samples);
    if (fspt->n_samples) {
        stats->min_samples_leaves_p =
            ((double) stats->min_samples_leaves) / fspt->n_samples;
        stats->max_samples_leaves_p =
            ((double) stats->max_samples_leaves) / fspt->n_samples;
        stats->median_samples_leaves_p =
            ((double) stats->median_samples_leaves) / fspt->n_samples;
        stats->first_quartile_samples_leaves_p =
            ((double) stats->first_quartile_samples_leaves) / fspt->n_samples;
        stats->third_quartile_samples_leaves_p =
            ((double) stats->third_quartile_samples_leaves) / fspt->n_samples;
    } else {
        stats->min_samples_leaves_p = 0;
        stats->max_samples_leaves_p = 0;
        stats->median_samples_leaves_p = 0;
        stats->first_quartile_samples_leaves_p = 0;
        stats->third_quartile_samples_leaves_p = 0;
    }

    /** Depth statistics **/
    qsort(leaves_array, leaves->size, sizeof(fspt_node *), cmp_depth_nodes);
    stats->max_depth = fspt->c_args ? fspt->c_args->max_depth : 0;
    stats->depth = fspt->depth;
    stats->min_depth_leaves = leaves_array[0]->depth;
    stats->median_depth_leaves =
        median((const void *)leaves_array, leaves->size, sizeof(fspt_node *),
                acc_depth);
    stats->first_quartile_depth_leaves =
        first_quartile((const void *)leaves_array, leaves->size,
                sizeof(fspt_node *), acc_depth);
    stats->third_quartile_depth_leaves =
        third_quartile((const void *)leaves_array, leaves->size,
                sizeof(fspt_node *), acc_depth);
    stats->min_depth_leaves_p =
        ((double) stats->min_depth_leaves) / fspt->depth;
    stats->median_depth_leaves_p =
        ((double) stats->median_depth_leaves) / fspt->depth;
    stats->first_quartile_depth_leaves_p =
        ((double) stats->first_quartile_depth_leaves) / fspt->depth;
    stats->third_quartile_depth_leaves_p =
        ((double) stats->third_quartile_depth_leaves) / fspt->depth;
    stats->balanced_index = 1. - ((double) (2. * fspt->depth - 1.)) / n_nodes;

    /** Node type statistics **/
    stats->n_leaves = leaves->size;
    stats->n_inner = inner_nodes->size;
    stats->n_leaves_p = ((double) leaves->size) / n_nodes;
    stats->n_inner_p = ((double) inner_nodes->size) / n_nodes;

    /** Split statistics **/
    for (int feat = 0; feat < n_features; ++feat) {
        int n = inner_nodes_by_split_feat[feat]->size;
        stats->split_features_count[feat] = n;
        stats->split_features_count_p[feat] =
            stats->n_inner ? ((double) n) / stats->n_inner : 0;
        if (!n) continue;
        fspt_node **nodes_on_feat = inner_nodes_by_split_feat_arrays[feat];
        qsort(nodes_on_feat, n, sizeof(fspt_node *), cmp_split_value_nodes);
        stats->min_split_values[feat] = nodes_on_feat[0]->split_value;
        stats->max_split_values[feat] = nodes_on_feat[n - 1]->split_value;
        stats->median_split_values[feat] =
            median((const void *)nodes_on_feat, n, sizeof(fspt_node *),
                    acc_split_value);
        stats->first_quartile_split_values[feat] =
            first_quartile((const void *)nodes_on_feat, n,
                    sizeof(fspt_node *), acc_split_value);
        stats->third_quartile_split_values[feat] =
            third_quartile((const void *)nodes_on_feat, n,
                    sizeof(fspt_node *), acc_split_value);
    }
    
    /** Score statistics **/
    qsort(leaves_array, leaves->size, sizeof(fspt_node *), cmp_score_nodes);
    stats->min_score = leaves_array[0]->score;
    stats->max_score = leaves_array[leaves->size - 1]->score;
    stats->median_score =
        median((const void *)leaves_array, leaves->size, sizeof(fspt_node *),
                acc_score);
    stats->first_quartile_score =
        first_quartile((const void *)leaves_array, leaves->size,
                sizeof(fspt_node *), acc_score);
    stats->third_quartile_score =
        third_quartile((const void *)leaves_array, leaves->size,
                sizeof(fspt_node *), acc_score);

    /** Free **/
free_arrays:
    if (leaves_array) free(leaves_array);
    if (inner_nodes_array) free(inner_nodes_array);
    for (int feat = 0; feat < n_features; ++feat) {
        if (inner_nodes_by_split_feat_arrays[feat])
            free(inner_nodes_by_split_feat_arrays[feat]);
    }
    if (inner_nodes_by_split_feat_arrays)
        free(inner_nodes_by_split_feat_arrays);
free_no_leaves:
    free_list(nodes);
    if (nodes_array) free(nodes_array);
    free_list(leaves);
    free_list(inner_nodes);
    for (int feat = 0; feat < n_features; ++feat) {
        free_list(inner_nodes_by_split_feat[feat]);
    }

    return stats;
}

static char *cause_to_string(NON_SPLIT_CAUSE c) {
    switch (c) {
        case SPLIT:
            return "   SPLIT    ";
            break;
        case UNKNOWN_CAUSE:
            return "   UNKNOW   ";
            break;
        case MAX_DEPTH:
            return " MAX_DEPTH  ";
            break;
        case MIN_SAMPLES:
            return "MIN_SAMPLES ";
            break;
        case MIN_VOLUME:
            return " MIN_VOLUME ";
            break;
        case MIN_LENGTH:
            return " MIN_LENGTH ";
            break;
        case MAX_COUNT:
            return " MAX_COUNT  ";
            break;
        case NO_SAMPLE:
            return " NO_SAMPLE  ";
            break;
        case MERGE:
            return "   MERGE    ";
            break;
        case UNIFORMITY:
            return " UNIFORMITY ";
            break;
        default:
            return "????????????";
    }
}


void export_score_data(FILE *stream, fspt_stats *s) {
    for (size_t i = 0; i < s->n_leaves; ++i) {
        fprintf(stream, "\
"LINTFORMAT" "FLT_FORMAT" "FLT_FORMAT" "FLT_FORMAT" "LINTFORMAT" "FLT_FORMAT" "FLT_FORMAT" "FLT_FORMAT" "INT_FORMAT" %s\n",
            i,
            s->score_vol_n_array[i].score,
            s->score_vol_n_array[i].volume_p,
            pow(s->score_vol_n_array[i].volume_p, 1. / s->fspt->n_features),
            s->score_vol_n_array[i].n_samples,
            (double) s->score_vol_n_array[i].n_samples / s->n_samples,
            (double) s->score_vol_n_array[i].n_samples
                / (s->score_vol_n_array[i].volume_p * s->volume),
            (double) s->score_vol_n_array[i].n_samples
                / s->score_vol_n_array[i].volume_p / s->n_samples,
            s->score_vol_n_array[i].cause,
            cause_to_string(s->score_vol_n_array[i].cause));
    }
}

void print_fspt_stats(FILE *stream, fspt_stats *s, char * title) {
    /** Title **/
    if (title) {
        int len = strlen(title);
        fprintf(stream, "      ╔═");
        for (int i = 0; i < len; ++ i) fprintf(stream, "═");
        fprintf(stream, "═╗\n");
        fprintf(stream, "      ║ %s ║\n", title);
        fprintf(stream, "      ╚═");
        for (int i = 0; i < len; ++ i) fprintf(stream, "═");
        fprintf(stream, "═╝\n\n");
    }
    if (!s) {
        fprintf(stream, "No fspt statistics.\n");
        return;
    }

    /** Node Types **/
    fprintf(stream, "         ┌─────────────────────────┐\n");
    fprintf(stream, "         │     NODE TYPE STATS     │\n");
    fprintf(stream, "         ├────────────┬────────────┤\n");
    fprintf(stream, "         │   leaves   │   inner    │\n");
    fprintf(stream, "┌────────┼────────────┼────────────┤\n");
    fprintf(stream, "│ n_nodes│"LINTFORMAT"│"LINTFORMAT"│\n",
            s->n_leaves,s->n_inner);
    fprintf(stream, "│relative│"FLT_FORMAT"│"FLT_FORMAT"│\n", s->n_leaves_p,
            s->n_inner_p);
    fprintf(stream, "└────────┴────────────┴────────────┘\n");

    /** Statistics **/
    int n_features = s->fspt->n_features;
    fprintf(stream,"\
                 ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐\n\
                 │   total    │    mean    │    min     │     Q1     │   median   │     Q3     │    max     │\n\
┌────┬───────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│ AB │  volume   │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│\n\
│ SO │ n_samples │"LINTFORMAT"│"FLT_FORMAT"│"LINTFORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"LINTFORMAT"│\n\
│ LU │   depth   │"INT_FORMAT"│"FLT_FORMAT"│"INT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"INT_FORMAT"│\n\
├────┼───────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│ RE │  volume   │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│\n\
│ LA │mean_length│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│\n\
│ TI │ n_samples │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│\n\
│ VE │   depth   │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│\n\
├────┼───────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n\
│    │   score   │     //     │"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│\n\
└────┴───────────┼────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┤\n\
                 │  min_samples parameter = "LEFTINTFOR"  min_volume_p parameter = "LEFTFLTFOR"             │\n\
                 └──────────────────────────────────────────────────────────────────────────────────────────┘\n\n",
        // absolute volume
        s->leaves_volume, s->mean_volume, s->min_volume,
        s->first_quartile_volume, s->median_volume, s->third_quartile_volume,
        s->max_volume,
        // absolute n_samples
        s->n_samples, s->mean_samples_leaves, s->min_samples_leaves,
        s->first_quartile_samples_leaves, s->median_samples_leaves,
        s->third_quartile_samples_leaves, s->max_samples_leaves,
        // absolute depth
        s->depth, s->mean_depth_leaves, s->min_depth_leaves,
        s->first_quartile_depth_leaves, s->median_depth_leaves, 
        s->third_quartile_depth_leaves, s->depth,
        // relative volume
        s->leaves_volume_p, s->mean_volume_p, s->min_volume_p,
        s->first_quartile_volume_p, s->median_volume_p,
        s->third_quartile_volume_p, s->max_volume_p,
        // relative mean length
        pow(s->leaves_volume_p, 1. / n_features),
        pow(s->mean_volume_p, 1. / n_features),
        pow(s->min_volume_p, 1. / n_features),
        pow(s->first_quartile_volume_p, 1. / n_features),
        pow(s->median_volume_p, 1. / n_features),
        pow(s->third_quartile_volume_p, 1. / n_features),
        pow(s->max_volume_p, 1. / n_features),
        // relative n_samples
        1.f, s->mean_samples_leaves_p, s->min_samples_leaves_p,
        s->first_quartile_samples_leaves_p, s->median_samples_leaves_p,
        s->third_quartile_samples_leaves_p, s->max_samples_leaves_p,
        // relative depth
        1.f, s->mean_depth_leaves_p, s->min_depth_leaves_p,
        s->first_quartile_depth_leaves_p, s->median_depth_leaves_p,
        s->third_quartile_depth_leaves_p, 1.f,
        // aboslute score
        s->mean_score, s->min_score, s->first_quartile_score, s->median_score,
        s->third_quartile_score, s->max_score,
        // parameters
        s->min_samples_param,
        s->min_volume_parameter);

    /** Thresholds **/
    fprintf(stream,"\
┌────────────┬──────────────────────────────────────┬───────────────────────────────────────────────────┐\n\
│            │ absolute values above fspt thresholds│     relative values above fspt thresholds         │\n\
│    fspt    ├────────────┬────────────┬────────────┼────────────┬────────────┬────────────┬────────────┤\n\
│  threshold │   volume   │ n_samples  │  n_leaves  │   volume   │mean_length │ n_samples  │  n_leaves  │\n\
├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    for (int i = 0; i < s->n_thresh && i < 100; ++i) {
        fprintf(stream,"\
│"FLT_FORMAT"│"FLT_FORMAT"│"LINTFORMAT"│"LINTFORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│\n",
            s->fspt_thresh[i],
            s->volume_above_thresh[i],
            s->n_samples_above_thresh[i],
            s->n_leaves_above_thresh[i],
            s->volume_above_thresh_p[i],
            pow(s->volume_above_thresh_p[i], 1./n_features),
            s->n_samples_above_thresh_p[i],
            s->n_leaves_above_thresh_p[i]);
    }
    fprintf(stream,"\
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n\n");

    /** Score **/
    fprintf(stream, "\
┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐\n\
│   score    │   volume   │mean length │ n_samples  │ n_samples  │  density   │  density   │ uniformity │   cause    │\n\
│    leaf    │ proportion │   in the   │   in the   │ proportion │   of the   │ proportion │   score    │   of the   │\n\
│  (sorted)  │    leaf    │    leaf    │    leaf    │    leaf    │    leaf    │    leaf    │    leaf    │    end     │\n\
├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    for (size_t i = 0; i < s->n_leaves && i < 100 ; ++i) {
        fprintf(stream, "\
│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"LINTFORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"STR_FORMAT"│\n",
            s->score_vol_n_array[i].score, s->score_vol_n_array[i].volume_p,
            pow(s->score_vol_n_array[i].volume_p, 1. / n_features),
            s->score_vol_n_array[i].n_samples,
            safe_divd(s->score_vol_n_array[i].n_samples, s->n_samples),
            safe_divd(s->score_vol_n_array[i].n_samples,
                s->score_vol_n_array[i].volume_p * s->volume),
            safe_divd(s->score_vol_n_array[i].n_samples,
                s->score_vol_n_array[i].volume_p * s->n_samples),
            s->score_vol_n_array[i].unf_score,
            cause_to_string(s->score_vol_n_array[i].cause));
    }
    fprintf(stream, "\
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n\n");

    /** Depth **/
    fprintf(stream, "\
┌─────┬─────────────────────────────────────────────────────────────────────────┬─────┐\n\
│depth│100%%    75%%      50%%      25%%       0%%      25%%      50%%      75%%    100%%│ tot │\n\
├─────┼┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬┼─────┤\n");
    for (int d = 0; d < s->depth && d < 1000; ++d) {
        double prop = s->n_nodes_by_depth_p[d];
        const int half = 36;
        const int length = 2 * half + 1;
        int half_prop = floor(half * prop);
        char depth_string[length + 1];
        for (int i = 0; i < half - half_prop; ++i) depth_string[i] = ' ';
        for (int i = half - half_prop; i < half + half_prop + 1; ++i)
            depth_string[i] = '.';
        for (int i = half + half_prop + 1; i < length; ++i)
            depth_string[i] = ' ';
        depth_string[length] = '\0';
        fprintf(stream, "│%5d│%s│%-5ld│\n", d + 1, depth_string,
                s->n_nodes_by_depth[d]);
    }
    fprintf(stream, "\
├─────┼┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴┼─────┤\n\
│depth│100%%    75%%      50%%      25%%       0%%      25%%      50%%      75%%    100%%│ tot │\n\
└─────┴─────────────────────────────────────────────────────────────────────────┴─────┘\n\n");

    /** Splits **/
    fprintf(stream, "\
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐\n\
│                                          SPLIT STATISTICS          (missing features means no split on it) │\n\
├────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┤\n\
│feat│   count    │  count_p   │    mean    │    min     │     Q1     │   median   │     Q3     │    max     │\n\
├────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    for (int feat = 0; feat < s->fspt->n_features && feat < 1000; ++feat) {
        if (!s->split_features_count[feat]) continue;
        fprintf(stream, "\
│% 4d│"INT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│"FLT_FORMAT"│\n",
            feat, s->split_features_count[feat], s->split_features_count_p[feat],
            s->mean_split_values[feat], s->min_split_values[feat],
            s->first_quartile_split_values[feat], s->median_split_values[feat],
            s->third_quartile_split_values[feat], s->max_split_values[feat]);
    }
    fprintf(stream, "\
└────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n\n");
}

void free_fspt_stats(fspt_stats *stats) {
    /* Volume */
    if (stats->volume_above_thresh)
        free(stats->volume_above_thresh);
    if (stats->volume_above_thresh_p)
        free(stats->volume_above_thresh_p);
    /* Samples */
    if (stats->n_samples_above_thresh)
        free(stats->n_samples_above_thresh);
    if (stats->n_samples_above_thresh_p)
        free(stats->n_samples_above_thresh_p);
    /* Nodes */
    if (stats->n_leaves_above_thresh)
        free(stats->n_leaves_above_thresh);
    if (stats->n_leaves_above_thresh_p)
        free(stats->n_leaves_above_thresh_p);
    if (stats->score_vol_n_array)
        free(stats->score_vol_n_array);
    /* Depth */
    if (stats->n_nodes_by_depth)
        free(stats->n_nodes_by_depth);
    if (stats->n_nodes_by_depth_p)
        free(stats->n_nodes_by_depth_p);
    /* Splits */
    if (stats->split_features_count)
        free(stats->split_features_count);
    if (stats->split_features_count_p)
        free(stats->split_features_count_p);
    if (stats->min_split_values)
        free(stats->min_split_values);
    if (stats->max_split_values)
        free(stats->max_split_values);
    if (stats->mean_split_values)
        free(stats->mean_split_values);
    if (stats->median_split_values)
        free(stats->median_split_values);
    if (stats->first_quartile_split_values)
        free(stats->first_quartile_split_values);
    if (stats->third_quartile_split_values)
        free(stats->third_quartile_split_values);
    /* Thresholds */
    if (stats->fspt_thresh)
        free(stats->fspt_thresh);
    /* Stats */
    free(stats);
}

void fspt_decision_func(size_t n, const fspt_t *fspt, const float *X,
                        fspt_node **nodes) {
    int n_features = fspt->n_features;
    for (size_t i = 0; i < n; i++) {
        const float *x = X + i * n_features;
        fspt_node *tmp_node = fspt->root;
        if (!tmp_node) {
            nodes[i] = NULL;
            continue;
        }
        while (tmp_node->type != LEAF) {
            int split_feature = tmp_node->split_feature;
            if (x[split_feature] <= tmp_node->split_value) {
                tmp_node = tmp_node->left;
            } else {
                tmp_node = tmp_node->right;
            }
        }
        if (tmp_node->type == LEAF) {
            nodes[i] = tmp_node;
        }
    }
}

void free_fspt_nodes(fspt_node *node) {
    if (!node) return;
    free_fspt_nodes(node->right);
    free_fspt_nodes(node->left);
    free(node);
}

void free_fspt(fspt_t *fspt) {
    if (!fspt) return;
    if (fspt->feature_limit) free((float *) fspt->feature_limit);
    if (fspt->feature_importance) free((float *) fspt->feature_importance);
    free_fspt_nodes(fspt->root);
    if (fspt->samples) free(fspt->samples);
    free(fspt);
}

/**
 * Propagates the count of the node `node` to the parents recursively.
 * If the node has a non nul count, it takes the minimum count of his
 * children.
 *
 * \param fspt The fspt.
 * \param node The node from where we propagate the counts. Must be a INNER
 *             node. Note the count of this node and his children must be set
 *             by the caller.
 */
static void propagate_count(fspt_t *fspt, fspt_node *node) {
    debug_assert(node->type == INNER);
    int old_count = node->count;
    if (node->count) {
        node->count = node->right->count < node->left->count ?
            node->right->count : node->left->count;
    }
    if ((old_count != node->count) && node->parent)
        propagate_count(fspt, node->parent);
}

/**
 * Helper function to merge subtrees with a non nul count. We assume that the
 * counts are correctly computed i.e. that all the subtrees with non nul count
 * have increasing count value.
 * This function does not update the fspt (like depth or n_nodes);
 *
 * \param node The node of the subtree to recursively merge if needed.
 */
static void recursive_merge_nodes(fspt_node *node) {
    if (!node) return;
    if (node->count) {
        free_fspt_nodes(node->right);
        free_fspt_nodes(node->left);
        node->right = NULL;
        node->left = NULL;
        node->type = LEAF;
        node->cause = MERGE;
        node->split_feature = 0;
        node->split_value = 0.f;
    } else {
        recursive_merge_nodes(node->right);
        recursive_merge_nodes(node->left);
    }
}

/**
 * Merges the subtrees with a non nul count. We assume that the
 * counts are correctly computed i.e. that all the subtrees with non nul count
 * have increasing count value.
 *
 * \param fspt The fspt to merge.
 */
static void merge_nodes(fspt_t *fspt) {
    recursive_merge_nodes(fspt->root);
    list *nodes = fspt_nodes_to_list(fspt, PRE_ORDER);
    fspt->n_nodes = nodes->size;
    fspt_node *current_node;
    int depth = 0;
    while ((current_node = (fspt_node *) list_pop(nodes))) {
        if (current_node->depth > depth) depth = current_node->depth;
    }
    fspt->depth = depth;
}

void fspt_predict(size_t n, const fspt_t *fspt, const float *X, float *Y) {
    fspt_node **nodes = malloc(n * sizeof(fspt_node *));
    fspt_decision_func(n, fspt, X, nodes);
    for (size_t i = 0; i < n; i++) {
        if (nodes[i] == NULL) {
            Y[i] = 0.;
        } else {
            debug_print("predicted score %f", nodes[i]->score);
            Y[i] = nodes[i]->score;
        }
    }
    free(nodes);
}

void fspt_rescore(fspt_t *fspt, score_args *s_args) {
    s_args->fspt = fspt;
    fspt->s_args = s_args;
    /* discover score args */
    s_args->discover = 1;
    fspt->score(s_args);
    /* score */
    list *leaves = fspt_leaves_to_list(fspt, PRE_ORDER);
    s_args->n_leaves = leaves->size;
    fspt_node **leaves_array = (fspt_node **) list_to_array(leaves);
    score_vol_n *score_vol_n_array = NULL;;
    if (s_args->need_normalize)
        score_vol_n_array = calloc(leaves->size, sizeof(score_vol_n));
    for (int i = 0; i < leaves->size; ++i) {
        fspt_node *leaf = leaves_array[i];
        s_args->node = leaf;
        leaf->score = fspt->score(s_args);
        if (s_args->need_normalize) {
            score_vol_n_array[i] =
                (score_vol_n) {
                    leaf->score,
                    leaf->volume / fspt->volume,
                    leaf->n_samples,
                    leaf->cause,
                    0.   // TODO: compute uniformity score
                };
        }
    }
    /* normalize */
    if (s_args->need_normalize) {
        s_args->normalize_pass = 1;
        qsort(score_vol_n_array, leaves->size, sizeof(score_vol_n),
                cmp_score_vol_n);
        s_args->score_vol_n_array = score_vol_n_array;
        for (int i = 0; i < leaves->size; ++i) {
            s_args->node = leaves_array[i];
            leaves_array[i]->score = fspt->score(s_args);
        }
    }
    free(score_vol_n_array);
    free(leaves_array);
    free_list(leaves);
}

void fspt_fit(size_t n_samples, float *X, criterion_args *c_args,
        score_args *s_args, fspt_t *fspt) {
    c_args->fspt = fspt;
    s_args->fspt = fspt;
    // TODO: what to do to have no double free ?
    //if (fspt->root)
    //   free_fspt_nodes(fspt->root);
    /* Builds the root */
    fspt_node *root = calloc(1, sizeof(fspt_node));
    root->type = LEAF;
    root->n_features = fspt->n_features;
    root->n_samples = n_samples;
    root->n_empty = n_samples; //Arbitray initialize s.t. Density=0.5
    root->samples = X;
    root->depth = 1;
    root->fspt = fspt;
    root->parent = NULL;
    root->volume = fspt->volume;
    /* Update fspt */
    fspt->n_nodes = 1;
    fspt->n_samples = n_samples;
    fspt->samples = X;
    fspt->root = root;
    fspt->depth = 1;
    fspt->c_args = c_args;
    fspt->s_args = s_args;
    /* discover score args */
    s_args->discover = 1;
    fspt->score(s_args);
    if (c_args->merge_nodes || s_args->need_normalize)
        s_args->score_during_fit = 0;
    if (s_args->score_during_fit) {
        s_args->node = root;
        root->score = fspt->score(s_args);
    }
    if (!n_samples) return;

    list *fifo = make_list(); // fifo of the nodes to examine
    list_insert(fifo, (void *)root);
    while (fifo->size > 0) {
        fspt_node *current_node = (fspt_node *) list_pop(fifo);
        c_args->node = current_node;
        fspt->criterion(c_args);
        if (c_args->forbidden_split) {
            debug_print(
                    "forbidden split node %p at depth %d and n_samples = %ld",
                    current_node, current_node->depth,
                    current_node->n_samples);
            if (s_args->score_during_fit) {
                s_args->node = current_node;
                current_node->score = fspt->score(s_args);
            }
        } else {
            fspt_node *left = calloc(1, sizeof(fspt_node));
            fspt_node *right = calloc(1, sizeof(fspt_node));
            fspt_split(fspt, current_node, c_args->best_index,
                    c_args->best_split, left, right);
            if (c_args->increment_count) {
                ++current_node->count;
                left->count = current_node->count;
                right->count = current_node->count;
            }
            propagate_count(fspt, current_node);
            list_insert_front(fifo, right);
            list_insert_front(fifo, left);
        }
    }
    free_list(fifo);
    if (c_args->merge_nodes) {
        merge_nodes(fspt);
    }
    list *leaves = fspt_leaves_to_list(fspt, PRE_ORDER);
    s_args->n_leaves = leaves->size;
    fspt_node **leaves_array = (fspt_node **) list_to_array(leaves);
    score_vol_n *score_vol_n_array = NULL;;
    if (s_args->need_normalize)
        score_vol_n_array = calloc(leaves->size, sizeof(score_vol_n));
    if (!s_args->score_during_fit) {
        for (int i = 0; i < leaves->size; ++i) {
            fspt_node *leaf = leaves_array[i];
            s_args->node = leaf;
            leaf->score = fspt->score(s_args);
            if (s_args->need_normalize) {
                score_vol_n_array[i] =
                    (score_vol_n) {
                        leaf->score,
                        leaf->volume / fspt->volume,
                        leaf->n_samples,
                        leaf->cause,
                        0. //TODO: compute uniformity score
                    };
            }
        }
    }
    if (s_args->need_normalize) {
        s_args->normalize_pass = 1;
        qsort(score_vol_n_array, leaves->size, sizeof(score_vol_n),
                cmp_score_vol_n);
        s_args->score_vol_n_array = score_vol_n_array;
        for (int i = 0; i < leaves->size; ++i) {
            s_args->node = leaves_array[i];
            leaves_array[i]->score = fspt->score(s_args);
        }
    }
    free(score_vol_n_array);
    free(leaves_array);
    free_list(leaves);
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
    /* save criterion_args */
    save_criterion_args_file(fp, fspt.c_args, succ);
    /* save score_args */
    save_score_args_file(fp, fspt.s_args, succ);
    /* save others */
    *succ &= fwrite(&fspt.n_nodes, sizeof(size_t), 1, fp);
    *succ &= fwrite(&fspt.n_samples, sizeof(size_t), 1, fp);
    *succ &= fwrite(&fspt.depth, sizeof(int), 1, fp);
    *succ &= fwrite(&fspt.count, sizeof(int), 1, fp);
    *succ &= fwrite(&fspt.volume, sizeof(double), 1, fp);
    /* to know if the file contains samples */
    *succ &= fwrite(&save_samples, sizeof(int), 1, fp);
    /* save samples if requested */
    size = fspt.n_samples * fspt.n_features;
    if (save_samples && size) {
        *succ &= (fwrite(fspt.samples, sizeof(float), size, fp) == size);
    }
    int have_root = 0;
    if (fspt.root) {
        have_root = 1;
        int version = NODE_VERSION;
        size_t size = sizeof(fspt_node);
        size_t n_nodes = fspt.n_nodes;
        *succ &= fwrite(&have_root, sizeof(int), 1, fp);
        *succ &= fwrite(&version, sizeof(int), 1, fp);
        *succ &= fwrite(&size, sizeof(size_t), 1, fp);
        *succ &= fwrite(&n_nodes, sizeof(size_t), 1, fp);
        pre_order_node_save(fp, *fspt.root, succ);
    } else {
        *succ &= fwrite(&have_root, sizeof(int), 1, fp);
    }
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
static fspt_node * pre_order_node_load(FILE *fp, size_t n_samples,
        float *samples, fspt_node *parent, fspt_t * fspt, int *succ) {
    /* load node */
    fspt_node *node = malloc(sizeof(fspt_node));
    *succ &= fread(node, sizeof(fspt_node), 1, fp);
    if (!*succ) return NULL;
    /* point on samples */
    node->samples = samples;
    node->n_samples = n_samples;
    node->parent = parent;
    node->fspt = fspt;
    /* load children */
    float *samples_r = NULL;
    float *samples_l = NULL;
    size_t n_samples_r = 0;
    size_t n_samples_l = 0;
    if (node->left) {
        if (samples && n_samples) {
            size_t split_index = 0;
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
        node->left = pre_order_node_load(fp, n_samples_l, samples_l, node,
                fspt, succ);
    if (node->right)
        node->right = pre_order_node_load(fp, n_samples_r, samples_r, node,
                fspt, succ);
    return node;
}

void fspt_load_file(FILE *fp, fspt_t *fspt, int load_samples, int load_c_args,
        int load_s_args, int load_root, int *succ) {
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
    fspt->volume = volume(fspt->n_features, feature_limit);
    /* load feature_importance */
    size = fspt->n_features;
    float *feature_importance = malloc(size * sizeof(float));
    *succ &= (fread(feature_importance, sizeof(float), size, fp) == size);
    fspt->feature_importance = feature_importance;
    /* load criterion_args */
    criterion_args *c_args = load_criterion_args_file(fp, succ);
    if (load_c_args)
        fspt->c_args = c_args;
    /* load score_args */
    score_args *s_args = load_score_args_file(fp, succ);
    if (load_s_args)
        fspt->s_args = s_args;
    /* load others */
    *succ &= fread(&fspt->n_nodes, sizeof(size_t), 1, fp);
    *succ &= fread(&fspt->n_samples, sizeof(size_t), 1, fp);
    *succ &= fread(&fspt->depth, sizeof(int), 1, fp);
    *succ &= fread(&fspt->count, sizeof(int), 1, fp);
    *succ &= fread(&fspt->volume, sizeof(double), 1, fp);
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
            assert(samples);
            *succ &= (fread(samples, sizeof(float), size, fp) == size);
            fspt->samples = samples;
        } else {
            fseek(fp, size *sizeof(float), SEEK_CUR);
        }
    }
    /* load root */
    int have_root = 0;
    *succ &= fread(&have_root, sizeof(int), 1, fp);
    if (have_root) {
        int version = 0;
        size_t size = 0;
        size_t n_nodes = 0;
        *succ &= fread(&version, sizeof(int), 1, fp);
        *succ &= fread(&size, sizeof(size_t), 1, fp);
        *succ &= fread(&n_nodes, sizeof(size_t), 1, fp);
        if (load_root && version == NODE_VERSION
                && size == sizeof(fspt_node)) {
            fspt->root =
                pre_order_node_load(fp, fspt->n_samples, fspt->samples, NULL,
                        fspt, succ);
        } else {
            fseek(fp, size * n_nodes, SEEK_CUR);
            if (load_root)
                fprintf(stderr, "Nodes not load : wrong version (%d) or size\
(saved size = %ld and sizeof(fspt_node) = %ld).\n",
                        version, size, sizeof(fspt_node));
        }
    }
}

void fspt_load(const char *filename, fspt_t *fspt, int load_samples,
        int load_c_args, int load_s_args, int load_root, int *succ) {
    fprintf(stderr, "Loading fspt from %s\n", filename);
    *succ = 1;
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    fspt_load_file(fp, fspt, load_samples, load_c_args, load_s_args, load_root,
            succ);
    fclose(fp);
}

#undef N_THRESH_STATS_FSPT
#undef FLT_FORMAT
#undef LEFTFLTFOR
#undef STR_FORMAT
#undef INT_FORMAT
#undef LINTFORMAT
#undef LEFTINTFOR
#undef NODE_VERSION
