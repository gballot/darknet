#ifndef FSPT_SCORE_H
#define FSPT_SCORE_H

#include "fspt.h"

/**
 * maps the string names of the functions to the score functions.
 *
 * \param s The string name of a score function.
 * \return the corresponding score function if it exists or NULL.
 */
extern score_func string_to_fspt_score(char *s);

/**
 * A score function called "density".
 * This score is the density of samples in the node normalized by the
 * density of the whole fspt.
 *
 * \param args The score arguments. Should at least contain the node and
 *             the fspt.
 */
extern float density_score(score_args *args);

/**
 * Prints a score_args to a file.
 *
 * \param stream The output stream.
 * \param a The score_args.
 * \param title An optional title.
 */
extern void print_fspt_score_args(FILE *stream, score_args *a, char *title);

/**
 * An euristic score function called "euristic".
 * This score is :
 * \sum f_i \frac{ R^+ } { R^+ + E \frac{\Delta I(R)} {\Delta I} }
 * Where f_i is the feature importance of feature i, R^+ is the number
 * of training samples it the node, \Delta I(R) is the differance
 * between the feature limits of feature i in the node, \Delta I is the
 * differance between the feature limits of feature i in the fspt, 
 * and E is the ratio between training samples in the tree and number of
 * feature.
 *
 * \param args The score arguments. Should at least contain the node and
 *             the fspt.
 */
extern float euristic_score(score_args *args);

#endif /* FSPT_SCORE_H */
