#include "distance_to_boundary.h"

#include <assert.h>
#include <math.h>

//#include "kolmogorov.h"
#include "kolmogorov_smirnov_dist.h"
#include "utils.h"

static float dist_to_bound_cpu(int d, const float *x,
        const float *lim) {
    float min = x[0] - lim[0];
    for (int i = 0; i < d; ++i) {
        if (x[i] < (lim[2*i] + lim[2*i + 1]) / 2) {
            float tmp = x[i] - lim[2*i];
            if (tmp < min) min = tmp;
        } else {
            float tmp = lim[2*i + 1] - x[i];
            if (tmp < min) min = tmp;
        }
    }
    return min;
}

static float min_half_length(int d, const float *lim) {
    float min = (lim[1] - lim[0]) / 2;
    for (int i = 1; i < d; ++i) {
        float d = (lim[2*i + 1] - lim[2*i]) / 2;
        if (d < min) min = d;
    }
    return min;
}

static float max_dist(int d, const float *lim) {
    float max = 0.f;
    for (int i = 0; i < d; ++i) {
        float d = (lim[2*i + 1] - lim[2*i]) / 2;
        if (d > max) max = d;
    }
    return max;
}

static float *relative_depth_cpu(int d, int n, const float *X,
        const float *lim) {
    float *Y = calloc(n, sizeof(float));
    float R = max_dist(d, lim);
    assert(R);
    for (int i = 0; i < n; ++i) {
        const float *x = X + d * i;
        Y[i] = dist_to_bound_cpu(d, x, lim) / R;
    }
    return Y;
}

static float null_hypothesis_dist(int d, const float *lim, float R, float y) {
    float cum = 1.f;
    for (int i = 0; i < d; ++i) {
        float ki = R * 2 / (lim[2*i + 1] - lim[2*i]);
        debug_assert(0 <= ki && ki <= 1);
        cum *= 1.f - ki * y;
    }
    debug_assert(0.f <= cum && cum <= 1.f);
    return 1.f - cum;
}

static float KS_stat_cpu(int d, int n, const float *X, const float *lim) {
    float *depths = relative_depth_cpu(d, n, X, lim);
    qsort_float(n, depths);
    float R = min_half_length(d, lim);
    float sup = 0.f;
    for (int i = 0; i < n; ++i) {
        float empirical = (float) i / n;
        float theoretical = null_hypothesis_dist(d, lim, R, depths[i]);
        float diff = empirical - theoretical;
        diff = ABS(diff);
        //debug_print("i = %d, diff = %g, empirical = %g, theoretical = %g, depth = %g",
        //        i, diff, empirical, theoretical, depths[i]);
        if (diff > sup) sup = diff;
    }
    float empirical = 1.f;
    float theoretical = null_hypothesis_dist(d, lim, R, depths[n-1]);
    float diff = empirical - theoretical;
    diff = ABS(diff);
    //debug_print("i = %d, diff = %g, empirical = %g, theoretical = %g, depth = %g",
    //        n, diff, empirical, theoretical, depths[n-1]);
    free(depths);
    if (diff > sup) sup = diff;
    debug_assert(0.f <= sup && sup <= 1.f);
    debug_print("sup = %g", sup);
    return sup;
}

double dist_to_bound_test(int d, int n, const float *X, const float *lim) {
    if (n == 0) return 1.;
    if (n == 1) return 0.;
#ifndef GPU
    float KS_stat = KS_stat_cpu(d, n, X, lim);
    debug_print("n = %d, sup = %g, sup * sqrt(n) = %g", n, KS_stat, pow(n, 0.5) * KS_stat);
    double p_value = KSfbar(n, KS_stat);
    return p_value;
#else /* GPU */
    float KS_stat;
    if (n < 100)
        KS_stat = KS_stat_cpu(d, n, X, lim);
    else
        KS_stat = KS_stat_cpu(d, n, X, lim);
    debug_print("n = %d, sup = %g, sup * sqrt(n) = %g", n, KS_stat, pow(n, 0.5) * KS_stat);
    double p_value = KSfbar(n, KS_stat);
    return p_value;
#endif /* GPU */
}
