#include "fspt_score.h"

#include <stdlib.h>

#include "fspt.h"
#include "utils.h"

float euristic_score(const fspt_t *fspt,const fspt_node *node) {
    float E = fspt->n_samples / fspt->n_features;
    float cum = 0;
    for (int i = 0; i < fspt->n_features; ++i) {
        float d_feature_local = node->feature_limit[2*i + 1]
            - node->feature_limit[2*i];
        float d_feature_global = fspt->feature_limit[2*i + 1]
            - fspt->feature_limit[2*i];
        float c = E * d_feature_local / (node->n_samples * d_feature_global);
        cum += fspt->feature_importance[i] / (1. + c);
    }
    return cum;
}

score_func string_to_fspt_score(char *s) {
    if (strcmp(s, "euristic") == 0) {
        return euristic_score;
    } else {
        error("unknow score function");
    }
    return NULL;
}
