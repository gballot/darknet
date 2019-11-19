#include "fspt_score.h"

#include <assert.h>
#include <stdlib.h>

#include "fspt.h"
#include "list.h"
#include "math.h"
#include "utils.h"

float density_score(score_args *args) {
    fspt_node *node = args->node;
    fspt_t *fspt = args->fspt;
    if (args->discover) {
        args->discover = 0;
        assert(args->calibration_volume_p);
        double calibration_full_score =
            args->calibration_n_samples_p / args->calibration_volume_p;
        args->calibration_tau =
            - log(1. - args->calibration_score) / calibration_full_score;
        return 0.f;
    }
    if (fspt->n_samples == 0) return 0.f;
    float uncalibred_score = ((float) node->n_samples / fspt->n_samples)
        * ((float) fspt->volume / node->volume);
    float score = 1. - exp(- args->calibration_tau * uncalibred_score);
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
    return score;
}

void print_fspt_score_args(FILE *stream, score_args *a, char *title) {
    /** Title **/
    if (title) {
        int len = strlen(title);
        fprintf(stream, "      ╔═");
        for (int i = 0; i < len; ++ i) fprintf(stream, "═");
        fprintf(stream, "═╗\n");
        fprintf(stream, "      ║ %s ║\n", title);
        fprintf(stream, "      ╚═");
        for (int i = 0; i < len; ++ i) fprintf(stream, "═");
        fprintf(stream, "═╝\n");
    }
    fprintf(stream, "Fspt score parameter :\n\
        Messages to change fitting behaviour :\n\
    -score_during_fit = %d\n\
        Messages for all score functions :\n\
    -fspt = %p\n\
    -node = %p\n\
    -discover = %d\n\
    -need_normalize = %d\n\
    -normalize_pass = %d\n\
        Messages for euristic score :\n\
    -compute_euristic_hyperparam = %d\n\
    -euristic_hyperparam = %f\n\
        Message for density score :\n\
    -calibration_score = %f\n\
    -calibration_n_samples_p = %f\n\
    -calibration_volume_p = %f\n\
    -calibration_tau = %f\n",
    a->score_during_fit, a->fspt, a->node, a->discover, a->need_normalize,
    a->normalize_pass, a->compute_euristic_hyperparam, a->euristic_hyperparam,
    a->calibration_score, a->calibration_n_samples_p, a->calibration_volume_p,
    a->calibration_tau);
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

