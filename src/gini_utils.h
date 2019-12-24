#ifndef GINI_UTILS_H
#define GINI_UTILS_H

#include "fspt_criterion.h"

/**
 * The gini criterion function.
 * This criterion function from Toward Safe Machine Learning paper returns
 * the splits that maximize the gain in the gini index before and after the
 * potential split.
 *
 * \param args Input/Output parameter.
 */
extern void gini_criterion(criterion_args *args);

/**
 * Computes the probability that n samples from a uniform distribution
 * over [0,1], makes a gain in the gini index greater than t by splitting
 * at s.
 */
extern double proba_gain_inferior_to(double t, double s, int n);


#endif /* GINI_UTILS_H */
