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


#endif /* GINI_UTILS_H */
