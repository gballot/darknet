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
 * \param fspt The fspt.
 * \param node The node wich we want to give a score. Note that the score is
 *             note affected to the node by this function.
 */
extern float euristic_score(const fspt_t *fspt,const fspt_node *node);

#endif /* FSPT_SCORE_H */
