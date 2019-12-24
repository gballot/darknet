#include "gini_utils.h"

#include <math.h>
#include "distance_to_boundary.h"
#include "utils.h"

#ifndef DEBUG
#define unit_static static
#else /* DEBUG */
#define unit_static  
#endif

#define EPS 0.00001

/**
 * Computes P(A <= 1/n sum_{i=1}^n 1_{X_i <= s} <= B).
 * Where X_i are independant uniform probabilities over [0, 1].
 *
 * \param A The inferior bound.
 * \param B The superior bound.
 * \param n The number of X_i.
 * \param s The threshold for the X_i
 * \return The probability.
 */
static long double proba_uninform_count(long double A, long double B,
        int n, long double s) {
    if (s <= 0.) return 0.;
    if (s >= 1.) return 1.;
    long double p = 0.;
    A = constrain_long_double(0., 1., A);
    B = constrain_long_double(0., 1., B);
    /*
    if (n*A - floor(n*A) < 1E-10) {
        p = pow(1 - s, n - floor(n * B)) - pow(1 - s, n - floor(n * A) + 1);
    } else 
        p = pow(1 - s, n - floor(n * B)) - pow(1 - s, n - floor(n * A));
    */
    int to = floor(n * B);
    int from = (n*A - floor(n*A) <= 1E-12) ? floor(n*A) : ceil(n*A);
    for (int i = from; i <= to; ++i) {
        p += binomial(n, i) * powl(s, i) * powl(1. - s, n - i);
    }
    // sitch the computation to put the biggest exponents on the smallest
    // value between s and 1-s.
    //if (n - from > to && s < 0.5) {
    //     normal
    /*} else {
        p = 1.;
        s = 1. - s;
        for (int i = from; i <= to; ++i) {
            p -= binomial(i, n) * powl(s, i) * powl(1. - s, n - i);
        }
    }
    */
    return p / n;
}

double proba_gain_inferior_to(double t, double s, int n) {
    if (t <= 0.) return 0.;
    if (t >= 0.5) return 1.;
    // To avoid divisions by very low numbers. By symetry.
    //if (s > 0.9) return proba_gain_inferior_to(t, 1 - s, n);
    polynome_t poly = {0};
    poly.a = t + 0.5;
    poly.b = 2. * t * s - s - 2. * t;
    poly.c = 0.5 * s * (2. * (s - 2) * t + s); 
    solve_polynome(&poly);
    if (poly.delta < 0.) return 0.;
    long double A = poly.x1;
    long double B = poly.x2;
    //double A2 = (2.*t + s - 2.*t*s - 2.*pow(t*t - 2.*t*s*s + 2.*t*s, 0.5)) / (2.*t + 1.);
    //double B2 = (2.*t + s - 2.*t*s + 2.*pow(t*t - 2.*t*s*s + 2.*t*s, 0.5)) / (2.*t + 1.);
    debug_assert(A <= B);
    //fprintf(stderr, "A=%Lg, B=%Lg, A2=%g, B2=%g\n", A, B, A2, B2);
    //debug_assert(ABS(A - A2) < 10E-5);
    //debug_assert(ABS(B - B2) < 10E-5);
    long double p = proba_uninform_count(A, B, n, s);
    return p;
}

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
        ++cause->count_min_length_p_hit;
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

typedef struct split_args {
    int feat;
    float node_min;
    float node_max;
    float *X;
    criterion_args *c_args;
    forbidden_split_cause *cause;
    double *best_gain;
    float *best_split;
    int *forbidden_split;
    int multi_threads;
} split_args;

/**
 * Finds the best split point on feature feat.
 */
static void best_split_on_feature(float node_min, float node_max,
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

static void *fill_best_splits(void *args) {
    split_args *a = (split_args *)args;
    criterion_args *c_args = a->c_args;
    int feat = a->feat;
    float *X = a->X;
    size_t n_samples = c_args->node->n_samples;
    int n_features = c_args->fspt->n_features;
    size_t n_bins = 0;
    size_t *cdf = malloc(2 * n_samples * sizeof(size_t));
    float *bins = malloc(2 * n_samples * sizeof(float));
    if (a->multi_threads) {
        float *x = malloc(n_samples * sizeof(float));
        copy_cpu(n_samples, X + feat, n_features, x, 1);
        qsort_float(n_samples, x);
        hist(n_samples, 1, x, a->node_min, &n_bins,
                cdf, bins);
        free(x);
    } else {
        qsort_float_on_index(feat, n_samples, n_features, X);
        hist(n_samples, n_features, X + feat, a->node_min, &n_bins,
                cdf, bins);
    }
    if (n_bins < 1) {
        *a->best_gain = -1.;
        *a->best_split = 0.f;
        free(cdf);
        free(bins);
        free(a);
        return NULL;
    }
    int local_best_gain_index = 0;
    double local_best_gain = 0.;
    int local_forbidden_split = 1;
    best_split_on_feature(a->node_min, a->node_max, n_samples,
            c_args->node->n_empty, c_args->node->volume, c_args->min_samples,
            c_args->min_volume_p * c_args->fspt->volume, c_args->min_length_p, 
            c_args->max_tries_p, n_bins,
            bins, cdf, &local_best_gain, &local_best_gain_index,
            &local_forbidden_split, a->cause);

    if (!local_forbidden_split) {
        float fspt_min = c_args->fspt->feature_limit[2*feat];
        float fspt_max = c_args->fspt->feature_limit[2*feat + 1];
        double relative_length = (a->node_max - a->node_min)
            / (fspt_max - fspt_min);
        *a->best_gain = local_best_gain
            * c_args->fspt->feature_importance[feat]
            * relative_length;
        *a->best_split = bins[local_best_gain_index];
        *a->forbidden_split = 0;
    } else {
        *a->best_gain = -1.;
        *a->best_split = 0.f;
    }
    free(cdf);
    free(bins);
    free(a);
    return NULL;
}

void gini_criterion(criterion_args *args) {
    fspt_t *fspt = args->fspt;
    fspt_node *node = args->node;
    args->end_of_fitting = 0;
    if (node->n_samples == 0) {
        ++args->count_no_sample_hit;
        node->cause = NO_SAMPLE;
        args->forbidden_split = 1;
        return;
    }
    if (node->n_samples + node->n_empty < (size_t) 2 * args->min_samples) {
        ++args->count_min_samples_hit;
        node->cause = MIN_SAMPLES;
        args->forbidden_split = 1;
        return;
    }
    if (node->depth >= args->max_depth) {
        ++args->count_max_depth_hit;
        node->cause = MAX_DEPTH;
        args->forbidden_split = 1;
        return;
    }
    if (node->volume < 2 * args->min_volume_p * fspt->volume) {
        ++args->count_min_volume_p_hit;
        node->cause = MIN_VOLUME;
        args->forbidden_split = 1;
        return;
    }
    float *feature_limit = get_feature_limit(node);
    if (args->uniformity_test_level == ALLWAYS_TEST_UNIFORMITY) {
        double p_value;
        /*
        if (args->unf_score_thresh < 1.
                && node->n_samples > (size_t) fspt->n_features) {
            struct unf_options options = {0};
            // TODO: put the right options in order to keep the feature limits.
            unf_score = unf_test_float(&options, node->samples,
                    node->n_samples, node->n_features);
            debug_print("uniformity_score = %g", unf_score);
            if (unf_score > args->unf_score_thresh) {
                ++args->count_uniformity_hit;
                node->cause = UNIFORMITY;
                args->forbidden_split = 1;
                return;
            }
        } else {
            ++args->count_min_samples_hit;
            node->cause = MIN_SAMPLES;
            args->forbidden_split = 1;
            return;
        }
        */
        p_value = dist_to_bound_test(fspt->n_features, node->n_samples,
                node->samples, feature_limit);
        debug_print("p-value uniformity test = %g", p_value);
        if (p_value > args->unf_alpha) {
            ++args->count_uniformity_hit;
            node->cause = UNIFORMITY;
            args->forbidden_split = 1;
            return;
        }
    }
    if (!respect_min_lenght_p(fspt->n_features, fspt->feature_limit,
                feature_limit, args->min_length_p)) {
        ++args->count_min_length_p_hit;
        args->forbidden_split = 1;
        node->cause = MIN_LENGTH;
        free(feature_limit);
        return;
    }
    double *best_gains = malloc(fspt->n_features * sizeof(double));
    float *best_splits = malloc(fspt->n_features * sizeof(float));
    int *random_features = random_index_order(0, fspt->n_features);
    float *X = node->samples;
    int forbidden_split = 1;
    int max_features = floor(fspt->n_features * args->max_features_p);
    forbidden_split_cause *causes =
        calloc(max_features, sizeof(forbidden_split_cause));
    pthread_t *threads = NULL;
    if (args->multi_threads) {
        threads = calloc(max_features, sizeof(pthread_t));
    }
    for (int i = 0; i < max_features; ++i) {
        int feat = random_features[i];
        float node_min = feature_limit[2*feat];
        float node_max = feature_limit[2*feat + 1];
        split_args *sp_args = calloc(1, sizeof(split_args));
        sp_args->feat = feat;
        sp_args->node_min = node_min;
        sp_args->node_max = node_max;
        sp_args->X = X;
        sp_args->c_args = args;
        sp_args->cause = causes + i;
        sp_args->best_gain = best_gains + i;
        sp_args->best_split = best_splits + i;
        sp_args->forbidden_split = &forbidden_split;
        sp_args->multi_threads = args->multi_threads;
        if (args->multi_threads) {
            pthread_create(threads + i, 0, fill_best_splits, (void *)sp_args);
        } else {
            fill_best_splits((void *)sp_args);
        }
    }
    for (int i = max_features; i < fspt->n_features; ++i) {
        best_gains[i] = -1.;
        best_splits[i] = 0.f;
    }
    if (args->multi_threads) {
        for (int i = 0; i < max_features; ++i) {
            pthread_join(threads[i], 0);
        }
        free(threads);
    }
    if (forbidden_split) {
        determine_cause(max_features, causes, args);
        args->forbidden_split = 1;
    } else {
        int rand_idx = max_index_double(best_gains, fspt->n_features);
        args->best_index = random_features[rand_idx];
        double best_gain = best_gains[rand_idx];
        args->best_split = best_splits[rand_idx];
        args->forbidden_split = 0;
        if (args->uniformity_test_level != ALLWAYS_TEST_UNIFORMITY
                && best_gain < args->gini_gain_thresh) {
            double p_value = 0.;
            if (args->uniformity_test_level == MIXED_TEST_UNIFORMITY
                    && args->unf_alpha < 1.) {
                p_value = dist_to_bound_test(fspt->n_features,
                        node->n_samples, node->samples,
                        feature_limit);
                debug_print("p-value uniformity test = %g", p_value);
                /*
                struct unf_options options = {0};
                // TODO: put the right options in order to keep the feature limits.
                unf_score = unf_test_float(&options, node->samples,
                        node->n_samples, node->n_features);
                debug_print("uniformity_score = %g", unf_score);
                */
            }
            if ((args->uniformity_test_level == MIXED_TEST_UNIFORMITY
                        && p_value <= args->unf_alpha)
                    || args->uniformity_test_level != MIXED_TEST_UNIFORMITY) {
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
                    node->cause = MAX_COUNT;
                    args->forbidden_split = 1;
                }
            } else {
                ++args->count_uniformity_hit;
                node->cause = UNIFORMITY;
                args->forbidden_split = 1;
            }
        } else {
            debug_print("best_index=%d, best_split=%f, gain=%f",
                    args->best_index, args->best_split, best_gain);
            node->count = 0;
        }
    }
    free(causes);
    free(feature_limit);
    free(best_gains);
    free(best_splits);
    free(random_features);
}

#undef unit_static
#undef EPS
