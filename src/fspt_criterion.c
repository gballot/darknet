#include "fspt_criterion.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "fspt.h"
#include "utils.h"

#ifndef DEBUG
#define unit_static static
#else /* DEBUG */
#define unit_static  
#endif

#define EPS 0.00001
#define FLOAT_FORMAT__ "%-16g"
#define POINTER_FORMAT "%-16p"
#define INTEGER_FORMAT "%-16d"
#define LONG_INTFORMAT "%-16ld"
#define CRITERION_ARGS_VERSION 4

typedef struct {
    size_t count_max_depth_hit;
    size_t count_min_samples_hit;
    size_t count_min_volume_p_hit;
    size_t count_min_length_p_hit;
} forbidden_split_cause;


/**
 * Computes the Gini index of a two classes set.
 * 
 * \param x The number of elements of the first class.
 * \param y The number of elements of the second class.
 * \return The gini index. 2xy/(x+y)^2.
 */
static double gini(double x, double y)
{
    return 2. * x * y / ( (x + y)*(x + y) );
}

static int respect_min_lenght_p(int n_features, const float* fspt_lim,
        const float *node_lim, double min_length_p) {
    if (min_length_p == 0.) return 1;
    for (int i = 0; i < n_features; ++i) {
        float node_min = node_lim[2*i];
        float node_max = node_lim[2*i + 1];
        float fspt_min = fspt_lim[2*i];
        float fspt_max = fspt_lim[2*i + 1];
        double relative_length = (double) (node_max - node_min)
            / (fspt_max - fspt_min);
        if (relative_length < min_length_p) {
            return 0;
        }
    }
    return 1;
}

static void determine_cause(forbidden_split_cause cause,
        criterion_args *args) {
    size_t tab[4] = {
        cause.count_min_volume_p_hit,
        cause.count_max_depth_hit,
        cause.count_min_samples_hit,
        cause.count_min_length_p_hit
    };
    if (!tab[0] && !tab[1] && !tab[2] && !tab[3]) return;
    int i = max_index_size_t(tab, 4);
    switch (i) {
        case 0:
            ++args->count_min_volume_p_hit;
            break;
        case 1:
            ++args->count_max_depth_hit;
            break;
        case 2:
            ++args->count_min_samples_hit;
            break;
        case 3:
            ++args->count_min_length_p_hit;
            break;
        default: break;
    }
}

/**
 * Computes \hat G(R, I, s) = n^+ / n * G(R^+) + n^- / n * G(R^-).
 * with the notations from Toward Safe Machine Learning.
 */
static double gini_after_split(float min, float max, float s, size_t n_left,
        size_t n_right, size_t n_empty, double node_volume, int min_samples,
        double min_volume, double min_length_p, int *forbidden_split,
        forbidden_split_cause *cause) {
    *forbidden_split = 0;
    double l = max - min;
    if (l == 0.) {
        *forbidden_split = 1;
        return 1.;
    }
    double prop_left = ((double) (s - min)) / ((double)l);
    double prop_right = ((double) (max - s)) / ((double) l);
    double n_empty_left = n_empty * prop_left;
    double n_empty_right = n_empty * prop_right;
    double volume_left = node_volume * prop_left;
    double volume_right = node_volume * prop_right;
    if (n_empty_left + n_left < min_samples
            || n_empty_right + n_right < min_samples) {
        ++cause->count_min_samples_hit;
        *forbidden_split = 1;
    }
    if (volume_left < min_volume
            || volume_right < min_volume) {
        ++cause->count_min_volume_p_hit;
        *forbidden_split = 1;
    }
    if (prop_left < min_length_p
            || prop_right < min_length_p) {
        ++cause->count_min_volume_p_hit;
        *forbidden_split = 1;
    }
    if (*forbidden_split) return 1.;
    double gini_left = gini(n_empty_left, n_left);
    double gini_right = gini(n_empty_right, n_right);
    double total_left = n_left + n_empty_left;
    double total_right = n_right + n_empty_right;
    double total = total_right + total_left;
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
unit_static void hist(size_t n, size_t step, const float *X, float lower_bond,
                 size_t *n_bins, size_t *cdf, float *bins) {
    *n_bins = 0;
    size_t last_cdf = 0;
    /* Special case for X[0] */
    float x_0 = X[0];
    debug_assert(x_0 >= lower_bond);
    if (x_0 > lower_bond) {
        float eps = EPS;
        while(x_0 - eps < lower_bond) {
            eps /= 2;
        }
        if (eps > 0) {
            bins[0] = x_0 - eps;
            cdf[0] = last_cdf;
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
    for (size_t i = 1; i < n ; ++i) {
        float x = X[i*step];
        debug_assert(x >= last_x);
        if (x > last_x) {
            float eps = EPS;
            while(x - eps < last_x) {
                eps /= 2;
            }
            if (eps > 0) {
                bins[*n_bins] = x - eps;
                cdf[(*n_bins)++] = last_cdf;
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

/**
 * Finds the best split point on feature feat.
 */
static void best_split_on_feature(int feat, float node_min, float node_max,
        size_t n_samples, size_t n_empty, double node_volume, int min_samples,
        double min_volume, double min_length_p,
        float max_tries_p, size_t n_bins, const float *bins,
        const size_t *cdf, double *best_gain, int *best_index,
        int *forbidden_split, forbidden_split_cause *cause) {
    *forbidden_split = 1;
    int local_best_gain_index = -1;
    double local_best_gain = 0.;
    size_t *random_index = random_index_order_size_t(0, n_bins);
    size_t max_bins = floor(n_bins * max_tries_p);
    if (!max_bins) max_bins = 1;
    for (size_t j = 0; j < max_bins; ++j) {
        size_t index = random_index[j];
        float bin = bins[index];
        debug_assert((node_min <= bin) && (bin <= node_max));
        size_t n_left = cdf[index];
        size_t n_right = n_samples - cdf[index];
        int local_forbidden_split = 0;
        double score = gini_after_split(node_min, node_max, bin, n_left,
                n_right, n_empty, node_volume, min_samples, min_volume,
                min_length_p, &local_forbidden_split, cause);
        if (local_forbidden_split) continue;
        double tmp_gain = 0.5 - score;
        if (tmp_gain > local_best_gain) {
            local_best_gain = tmp_gain;
            local_best_gain_index = index;
            *forbidden_split = 0;
        }
    }
    free(random_index);
    *best_gain = local_best_gain;
    *best_index = local_best_gain_index;
}

void gini_criterion(criterion_args *args) {
    fspt_t *fspt = args->fspt;
    fspt_node *node = args->node;
    args->end_of_fitting = 0;
    if (node->n_samples == 0) {
        ++args->count_no_sample_hit;
        args->forbidden_split = 1;
        return;
    }
    if (node->n_samples + node->n_empty < (size_t) 2 * args->min_samples) {
        ++args->count_min_samples_hit;
        args->forbidden_split = 1;
        return;
    }
    if (node->depth >= args->max_depth) {
        ++args->count_max_depth_hit;
        args->forbidden_split = 1;
        return;
    }
    if (node->volume < 2 * args->min_volume_p * fspt->volume) {
        ++args->count_min_volume_p_hit;
        args->forbidden_split = 1;
        return;
    }
    float *feature_limit = get_feature_limit(node);
    if (!respect_min_lenght_p(fspt->n_features, fspt->feature_limit,
                feature_limit, args->min_length_p)) {
        ++args->count_min_length_p_hit;
        args->forbidden_split = 1;
        free(feature_limit);
        return;
    }
    forbidden_split_cause cause = {0};
    double *best_gains = malloc(fspt->n_features * sizeof(double));
    float *best_splits = malloc(fspt->n_features * sizeof(float));
    size_t *cdf = malloc(2 * node->n_samples * sizeof(size_t));
    float *bins = malloc(2 * node->n_samples * sizeof(float));
    int *random_features = random_index_order(0, fspt->n_features);
    float *X = node->samples;
    int forbidden_split = 1;
    int max_features = floor(fspt->n_features * args->max_features_p);
    for (int i = 0; i < max_features; ++i) {
        int feat = random_features[i];
        float node_min = feature_limit[2*feat];
        float node_max = feature_limit[2*feat + 1];
        size_t n_bins = 0;
        qsort_float_on_index(feat, node->n_samples, fspt->n_features, X);
        hist(node->n_samples, fspt->n_features, X + feat, node_min, &n_bins,
                cdf, bins);
        if (n_bins < 1) {
            best_gains[i] = -1.;
            best_splits[i] = 0.f;
            continue;
        }
        int local_best_gain_index = 0;
        double local_best_gain = 0.;
        int local_forbidden_split = 1;
        best_split_on_feature(feat, node_min, node_max, node->n_samples,
                node->n_empty, node->volume, args->min_samples,
                args->min_volume_p * fspt->volume, args->min_length_p, 
                args->max_tries_p, n_bins,
                bins, cdf, &local_best_gain, &local_best_gain_index,
                &local_forbidden_split, &cause);

        if (!local_forbidden_split) {
            float fspt_min = fspt->feature_limit[2*feat];
            float fspt_max = fspt->feature_limit[2*feat + 1];
            double relative_length = (node_max - node_min)
                / (fspt_max - fspt_min);
            best_gains[i] = local_best_gain * fspt->feature_importance[feat]
                * relative_length;
            best_splits[i] = bins[local_best_gain_index];
            forbidden_split = 0;
        } else {
            best_gains[i] = -1.;
            best_splits[i] = 0.f;
        }
    }
    for (int i = max_features; i < fspt->n_features; ++i) {
        best_gains[i] = -1.;
        best_splits[i] = 0.f;
    }
    free(bins);
    free(cdf);
    if (forbidden_split) {
        determine_cause(cause, args);
        args->forbidden_split = 1;
    } else {
        int rand_idx = max_index_double(best_gains, fspt->n_features);
        args->best_index = random_features[rand_idx];
        double best_gain = best_gains[rand_idx];
        args->best_split = best_splits[rand_idx];
        args->forbidden_split = 0;
        if (best_gain < args->gini_gain_thresh) {
            if (args->middle_split) {
                /* split in the middle of the largest feature */
                int new_index = 0;
                float new_best_split = 0.f;
                double max_dlim = 0.;
                for (int i = 0; i < fspt->n_features; ++i) {
                    float node_min = feature_limit[2*i];
                    float node_max = feature_limit[2*i + 1];
                    float fspt_min = fspt->feature_limit[2*i];
                    float fspt_max = fspt->feature_limit[2*i + 1];
                    double relative_length = (node_max - node_min)
                        / (fspt_max - fspt_min);
                    if (relative_length > max_dlim) {
                        max_dlim = relative_length;
                        new_index = i;
                        new_best_split = (node_max + node_min) / 2;
                    }
                }
                debug_print("new index, new best split = %d,%f",
                        new_index, new_best_split);
                args->best_index = new_index;
                args->best_split = new_best_split;
            }
            args->increment_count = 1;
            debug_print(
                    "gain thresh violation at depth %d and count %d, gain %f",
                    node->depth, node->parent ? node->parent->count : 0,
                    best_gain);
            if (node->count >= args->max_consecutive_gain_violations) {
                ++args->count_max_count_hit;
                args->forbidden_split = 1;
            }
        } else {
            debug_print("best_index=%d, best_split=%f, gain=%f",
                    args->best_index, args->best_split, best_gain);
            node->count = 0;
        }
    }
    free(feature_limit);
    free(best_gains);
    free(best_splits);
    free(random_features);
}

void print_fspt_criterion_args(FILE *stream, criterion_args *a, char *title) {
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
    fprintf(stream, "\
┌──────────────────────────────────────────────┐\n\
│           FSPT CRITERION ARGUMENTS           │\n\
├──────────────────────────────────────────────┤\n\
│     Messages to change fitting behaviour     │\n\
├─────────────────────────────┬────────────────┤\n\
│                 merge_nodes │"INTEGER_FORMAT"│\n\
├─────────────────────────────┴────────────────┤\n\
│   Messages for all the criterion functions   │\n\
├─────────────────────────────┬────────────────┤\n\
│                        fspt │"POINTER_FORMAT"│\n\
│                        node │"POINTER_FORMAT"│\n\
│                   max_depth │"INTEGER_FORMAT"│\n\
│         count_max_depth_hit │"LONG_INTFORMAT"│\n\
│                 min_samples │"INTEGER_FORMAT"│\n\
│       count_min_samples_hit │"LONG_INTFORMAT"│\n\
│                min_volume_p │"FLOAT_FORMAT__"│\n\
│      count_min_volume_p_hit │"LONG_INTFORMAT"│\n\
│                min_length_p │"FLOAT_FORMAT__"│\n\
│      count_min_length_p_hit │"LONG_INTFORMAT"│\n\
│         count_max_count_hit │"LONG_INTFORMAT"│\n\
│         count_no_sample_hit │"LONG_INTFORMAT"│\n\
│                  best_index │"INTEGER_FORMAT"│\n\
│                  best_split │"FLOAT_FORMAT__"│\n\
│             forbidden_split │"INTEGER_FORMAT"│\n\
│             increment_count │"INTEGER_FORMAT"│\n\
│              end_of_fitting │"INTEGER_FORMAT"│\n\
├─────────────────────────────┴────────────────┤\n\
│         Messages for gini_criterion          │\n\
├─────────────────────────────┬────────────────┤\n\
│                 max_tries_p │"FLOAT_FORMAT__"│\n\
│              max_features_p │"FLOAT_FORMAT__"│\n\
│            gini_gain_thresh │"FLOAT_FORMAT__"│\n\
│max_consecutive_gain_violati │"INTEGER_FORMAT"│\n\
│                middle_split │"INTEGER_FORMAT"│\n\
└─────────────────────────────┴────────────────┘\n\n",
    a->merge_nodes, a->fspt, a->node,
    a->max_depth, a->count_max_depth_hit,
    a->min_samples, a->count_min_samples_hit,
    a->min_volume_p, a->count_min_volume_p_hit,
    a->min_length_p, a->count_min_length_p_hit,
    a->count_max_count_hit,
    a->count_no_sample_hit,
    a->best_index, a->best_split, a->forbidden_split,
    a->increment_count, a->end_of_fitting, a->max_tries_p, a->max_features_p,
    a->gini_gain_thresh, a->max_consecutive_gain_violations, a->middle_split);
}

void save_criterion_args_file(FILE *fp, criterion_args *c, int *succ) {
    int contains_args = 0;
    int version = CRITERION_ARGS_VERSION;
    if (c) {
        contains_args = 1;
        size_t size = sizeof(criterion_args);
        *succ &= fwrite(&contains_args, sizeof(int), 1, fp);
        *succ &= fwrite(&version, sizeof(int), 1, fp);
        *succ &= fwrite(&size, sizeof(size_t), 1, fp);
        *succ &= fwrite(c, sizeof(criterion_args), 1, fp);
    } else {
        *succ &= fwrite(&contains_args, sizeof(int), 1, fp);
    }
}

criterion_args *load_criterion_args_file(FILE *fp, int *succ) {
    criterion_args *c = NULL;
    int contains_args = 0;
    int version = 0;
    size_t size = 0;
    *succ &= fread(&contains_args, sizeof(int), 1, fp);
    if (contains_args == 1) {
        *succ &= fread(&version, sizeof(int), 1, fp);
        *succ &= fread(&size, sizeof(size_t), 1, fp);
        if (version == CRITERION_ARGS_VERSION
                && size == sizeof(criterion_args)) {
            c = malloc(sizeof(criterion_args));
            *succ &= fread(c, sizeof(criterion_args), 1, fp);
        } else {
            fseek(fp, size, SEEK_CUR);
            fprintf(stderr, "Wrong criterion version (%d) or size\
(saved size = %ld and sizeof(criterion_args) = %ld).",
                    version, size, sizeof(score_args));
        }
    } else if (contains_args != 0) {
        fprintf(stderr, "ERROR : in load_criterion_args_file - contains_args = %d",
                contains_args);
    }
    return c;
}

criterion_func string_to_fspt_criterion(char *s) {
    if (strcmp(s, "gini") == 0) {
        return gini_criterion;
    } else {
        return NULL;
    }
}

#undef unit_static
#undef EPS
#undef FLOAT_FORMAT__
#undef POINTER_FORMAT
#undef INTEGER_FORMAT
#undef CRITERION_ARGS_VERSION
#undef LONG_INTFORMAT

