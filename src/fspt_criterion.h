#ifndef GINI_H
#define GINI_H

#include <stddef.h>

#include "fspt.h"

extern criterion_func string_to_fspt_criterion(char *s);
extern void gini_criterion(criterion_args *args);

#endif /* GINI_H */
