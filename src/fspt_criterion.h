#ifndef GINI_H
#define GINI_H

#include <stddef.h>

#include "fspt.h"

extern criterion_func string_to_fspt_criterion(char *s);
extern void gini_criterion(criterion_args *args);
#ifdef DEBUG
extern void hist(size_t n, size_t step, const float *X, float lower_bond,
                 size_t *n_bins, size_t *cdf, float *bins);
#endif

#endif /* GINI_H */
