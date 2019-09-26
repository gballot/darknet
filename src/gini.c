#include "gini.h"

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "fspt.h"
#include "utils.h"


#define EPS 0.00001

/**
 * Computes the Gini index of a two classes set.
 * 
 * \param x The number of elements of the first class.
 * \param y The number of elements of the second class.
 * \return The gini index. 2xy/(x+y)^2.
 */
static float gini(float x, float y)
{
    return 2 * x * y / ( (x + y)*(x + y) );
}


/**
 * Computes \hat G(R, I, s) = n^+ / n * G(R^+) + n^- / n * G(R^-).
 * with the notations from Toward Safe Machine Learning.
 */
static float gini_after_split(float min, float max, float s, size_t n_left,
        size_t n_right, int n_empty, int min_samples) {
    float l = max - min;
    if (l == 0.)
        return FLT_MAX;
    float n_empty_left = n_empty * (s - min) / l;
    float n_empty_right = n_empty * (max -s) / l;
    if (n_empty_left + n_left < min_samples
            || n_empty_right + n_right < min_samples)
        return FLT_MAX;
    //TODO
    float gini_left = gini(n_empty_left, n_left);
    float gini_right = gini(n_empty_right, n_right);
    int total_left = n_left + n_empty_left;
    int total_right = n_right + n_empty_right;
    int total = total_right + total_left;
    return gini_left * total_left / total + gini_right * total_right / total;
}

/**
 * Creates an hitogram of X with regards to the potential split points
 * depicted in the paper Toward Safe Machine Learning. That is to say,
 * {max(lower_bond, X[0]-EPS), X[0], X[step]-EPS, X[step],...,
 *  X[(n-1)*step]-EPS, X[(n-1)*step]}.
 *
 * Uses macro EPS.
 *
 * \param n The number of floats in X.
 * \param step The step to acces X : X[i*step].
 * \param X The float array. Is assumed sorted.
 * \param lower_bond Lower bond of X.
 * \param n_bins Output paramerter. the number of bins for the bins and cdf
 *               arrays.
 * \param cdf Output parameter. cdf[i] contains the cumulative number of
 *            elements <= bins[i].
 * \param bins Output parameter. Contains the elements in X and them minus EPS.
 */
static void hist(size_t n, size_t step, const float *X, float lower_bond,
                 size_t *n_bins, size_t *cdf, float *bins) {
    *n_bins = 0;
    size_t last_cdf = 0;
    /* Special case for X[0] */
    float x_0 = X[0];
    if (x_0 > lower_bond) {
        float eps = EPS;
        while(x_0 - eps < lower_bond) {
            eps /= 2;
        }
        if (eps > 0) {
            bins[0] = x_0 - eps;
            cdf[0] = ++last_cdf;
            bins[1] = x_0;
            cdf[1] = ++last_cdf;
            *n_bins = 2;
        } else {
            bins[0] = x_0;
            cdf[0] = ++last_cdf;
            *n_bins = 1;
        }
    } else {
        bins[0] = x_0;
        cdf[0] = ++last_cdf;
        *n_bins = 1;
    }
    /* build histogram */
    float last_x = x_0;
    for (size_t i = 0; i < n ; ++i) {
        float x = X[i*step];
        assert(x >= last_x);
        if (x > last_x) {
            float eps = EPS;
            while(x - eps < last_x) {
                eps /= 2;
            }
            if (eps > 0) {
                bins[*n_bins] = x - eps;
                cdf[(*n_bins)++] = ++last_cdf;
                bins[*n_bins] = x;
                cdf[(*n_bins)++] = ++last_cdf;
            } else {
                cdf[*n_bins - 1] = ++last_cdf;
            }
        } else {
            cdf[*n_bins - 1] = ++last_cdf;
        }
        last_x = x;
    }
}

void gini_criterion(criterion_args *args) {
    fspt_t *fspt = args->fspt;
    fspt_node *node = args->node;
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
        size_t *cdf = malloc(2 * fspt->n_features * sizeof(float));
        float *bins = malloc(2 * fspt->n_features * sizeof(float));
        size_t n_bins = 0;
        hist(node->n_samples, fspt->n_features, X + feat, node_min, &n_bins,
                cdf, bins);
        bins = realloc(bins, n_bins);
        cdf = realloc(cdf, n_bins);
        if (n_bins < 1) continue;
        size_t local_best_gain_index = 0;
        float local_best_gain = 0.;

        //TODO: could add max_try policy.
        //Add an argument 0<max_try_p<1
        //and take a subsample of bins of size floor(max(1,n_bins*max_try_p))
        for (size_t j = 0; j < n_bins; ++j) {
            // why ??
            assert(node->n_empty == node->n_samples);
            float bin = bins[j];
            size_t n_left = cdf[j];
            size_t n_right = node->n_samples - cdf[j];
            float score = gini_after_split(node_min, node_max, bin, n_left,
                    n_right, node->n_empty, fspt->min_samples);
            float tmp_gain = current_score - score;
            if (tmp_gain > local_best_gain) {
                local_best_gain = tmp_gain;
                local_best_gain_index = j;
            }
        }



        float fspt_min = fspt->feature_limit[2*feat];
        float fspt_max = fspt->feature_limit[2*feat + 1];
        float relative_length = (node_max - node_min) / (fspt_max - fspt_min);
        best_gains[feat] = local_best_gain * fspt->feature_importance[feat]
            * relative_length;
        best_splits[feat] = bins[local_best_gain_index];
    }
    int *best_feature_index = &args->best_index;
    float *best_gain = &args->gain;
    float *best_split = &args->best_split;
    *best_feature_index = max_index(best_gains, fspt->n_features);
    *best_gain = best_gains[*best_feature_index];
    *best_split = best_splits[*best_feature_index];
    if (*best_gain < 0) {
        *best_feature_index = FAIL_TO_FIND;
        debug_print("fail to find any split point ad depth %d and n_samples %d",
                node->depth, node->n_samples);
    } else if (*best_gain < args->thresh) {
        debug_print("fail to find any split point ad depth %d and count %d",
                node->depth, node->count);
        node->count += 1;
        int v = 10 > fspt->n_samples / 500 ? 10 : fspt->n_samples / 500;
        if (node->count >= v) {
            *best_feature_index = FAIL_TO_FIND;
        }
    } else {
        node->count = 0;
    }

    free(random_features);
}

#undef EPS
