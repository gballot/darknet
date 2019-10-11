#ifndef GINI_H
#define GINI_H

#include <stddef.h>

#include "fspt.h"

/**
 * Convert a string representing the name of a criterion function
 * to a pointer to this function.
 *
 * \param s the string name of hte criterion function.
 * \return A pointer to this criterion function if the name exists or
 *         NULL.
 */
extern criterion_func string_to_fspt_criterion(char *s);

/**
 * The gini criterion function.
 * This criterion function from Toward Safe Machine Learning paper returns
 * the splits that maximize the gain in the gini index before and after the
 * potential split.
 *
 * \param args Input/Output parameter.
 */
extern void gini_criterion(criterion_args *args);

#ifdef DEBUG
extern void hist(size_t n, size_t step, const float *X, float lower_bond,
                 size_t *n_bins, size_t *cdf, float *bins);
#endif

#endif /* GINI_H */
