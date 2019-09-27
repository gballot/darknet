#ifndef FSPT_SCORE_H
#define FSPT_SCORE_H

#include "fspt.h"

extern score_func string_to_fspt_score(char *s);
extern float euristic_score(const fspt_t *fspt,const fspt_node *node);

#endif /* FSPT_SCORE_H */
