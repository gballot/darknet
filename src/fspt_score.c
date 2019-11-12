#include "fspt_score.h"

#include <stdlib.h>

#include "fspt.h"
#include "list.h"
#include "utils.h"

float density_score(score_args *args) {
    fspt_node *node = args->node;
    fspt_t *fspt = args->fspt;
    if (args->discover) {
        args->discover = 0;
        args->need_normalize = 1;
        return 0.f;
    }
    if (args->normalize_pass) {
        return node->score / args->max_score;
    }
    float score =
        (node->n_samples / fspt->n_samples) * (fspt->volume / node->volume);
    if (score > args->max_score)
        args->max_score = score;
    return score;
}

float euristic_score(score_args *args) {
    fspt_node *node = args->node;
    fspt_t *fspt = args->fspt;
    if (args->discover) {
        if (!args->euristic_hyperparam)
            args->compute_euristic_hyperparam = 1;
        args->discover = 0;
        args->need_normalize = 0;
        return 0.f;
    }
    if (args->compute_euristic_hyperparam) {
        /* euristic parameter is the average number of samples per leaves */
        list *node_list = fspt_nodes_to_list(fspt, PRE_ORDER);
        fspt_node *n;
        int n_leaves = 0;
        while ((n = (fspt_node *) list_pop(node_list))) {
            if (n->type == LEAF) {
                ++n_leaves;
            }
        }
        args->euristic_hyperparam = ((float) fspt->n_samples) / n_leaves;
        free_list(node_list);
        args->compute_euristic_hyperparam = 0;
    }
    if (node->n_samples == 0) return 0.f;
    float E = args->euristic_hyperparam;
    float cum = 0;
    float cum2 = 0;
    float *feature_limit = get_feature_limit(node);
    for (int i = 0; i < fspt->n_features; ++i) {
        float d_feature_local = feature_limit[2*i + 1]
            - feature_limit[2*i];
        float d_feature_global = fspt->feature_limit[2*i + 1]
            - fspt->feature_limit[2*i];
        float c = E * d_feature_local / (node->n_samples * d_feature_global);
        cum += fspt->feature_importance[i] / (1. + c);
        cum2 += fspt->feature_importance[i];
    }
    free(feature_limit);
    float score = cum / cum2;
    if (score > args->max_score)
        args->max_score = score;
    return score;
}

score_func string_to_fspt_score(char *s) {
    if (strcmp(s, "euristic") == 0) {
        return euristic_score;
    }  else if (strcmp(s, "density") == 0) {
        return density_score;
    } else {
        return NULL;
    }
}

