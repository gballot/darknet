#include "fspt_score.h"

#include <assert.h>
#include <stdlib.h>

#include "fspt.h"
#include "list.h"
#include "math.h"
#include "utils.h"

#define FLOAT_FORMAT__ "%-16g"
#define POINTER_FORMAT "%-16p"
#define INTEGER_FORMAT "%-16d"

double density_score(score_args *args) {
    fspt_node *node = args->node;
    fspt_t *fspt = args->fspt;
    if (args->discover) {
        args->discover = 0;
        if (!args->calibration_volume_p && args->calibration_feat_length_p)
            args->calibration_volume_p =
                pow(args->calibration_feat_length_p, fspt->n_features);
        assert(args->calibration_volume_p);
        double calibration_full_score =
            args->calibration_n_samples_p /
            pow(args->calibration_volume_p, 1. - args->volume_penalization);
        args->calibration_tau =
            - log(1. - args->calibration_score) / calibration_full_score;
        args->need_normalize = 0;
        args->score_during_fit = 1;
        return 0.f;
    }
    if (fspt->n_samples == 0) return 0.f;
    double score = ((double) node->n_samples / fspt->n_samples)
        * pow(fspt->volume / node->volume, 1. - args->volume_penalization);
    if (args->exponential_normalization)
        score = 1. - exp(- args->calibration_tau * score);
    return score;
}

double euristic_score(score_args *args) {
    fspt_node *node = args->node;
    fspt_t *fspt = args->fspt;
    if (args->discover) {
        if (!args->euristic_hyperparam)
            args->compute_euristic_hyperparam = 1;
        args->discover = 0;
        args->need_normalize = 0;
        args->score_during_fit = 1;
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
        args->euristic_hyperparam = ((double) fspt->n_samples) / n_leaves;
        free_list(node_list);
        args->compute_euristic_hyperparam = 0;
    }
    if (node->n_samples == 0) return 0.f;
    double E = args->euristic_hyperparam;
    double cum = 0;
    double cum2 = 0;
    float *feature_limit = get_feature_limit(node);
    for (int i = 0; i < fspt->n_features; ++i) {
        double d_feature_local = feature_limit[2*i + 1]
            - feature_limit[2*i];
        double d_feature_global = fspt->feature_limit[2*i + 1]
            - fspt->feature_limit[2*i];
        double c = E * d_feature_local / (node->n_samples * d_feature_global);
        cum += fspt->feature_importance[i] / (1. + c);
        cum2 += fspt->feature_importance[i];
    }
    free(feature_limit);
    double score = cum / cum2;
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
        fprintf(stream, "═╝\n\n");
    }
    fprintf(stream, "\
┌──────────────────────────────────────────────┐\n\
│             FSPT SCORE ARGUMENTS             │\n\
├──────────────────────────────────────────────┤\n\
│     Messages to change fitting behaviour     │\n\
├─────────────────────────────┬────────────────┤\n\
│            score_during_fit │"INTEGER_FORMAT"│\n\
├─────────────────────────────┴────────────────┤\n\
│     Message for all the score functions      │\n\
├─────────────────────────────┬────────────────┤\n\
│                        fspt │"POINTER_FORMAT"│\n\
│                        node │"POINTER_FORMAT"│\n\
│                    discover │"INTEGER_FORMAT"│\n\
│              need_normalize │"INTEGER_FORMAT"│\n\
│              normalize_pass │"INTEGER_FORMAT"│\n\
├─────────────────────────────┴────────────────┤\n\
│         Messages for euristic_score          │\n\
├─────────────────────────────┬────────────────┤\n\
│ compute_euristic_hyperparam │"INTEGER_FORMAT"│\n\
│         euristic_hyperparam │"FLOAT_FORMAT__"│\n\
├─────────────────────────────┴────────────────┤\n\
│         Messages for density_score           │\n\
├─────────────────────────────┬────────────────┤\n\
│   exponential_normalization │"INTEGER_FORMAT"│\n\
│           calibration_score │"FLOAT_FORMAT__"│\n\
│     calibration_n_samples_p │"FLOAT_FORMAT__"│\n\
│        calibration_volume_p │"FLOAT_FORMAT__"│\n\
│   calibration_feat_length_p │"FLOAT_FORMAT__"│\n\
│         volume_penalization │"FLOAT_FORMAT__"│\n\
│             calibration_tau │"FLOAT_FORMAT__"│\n\
└─────────────────────────────┴────────────────┘\n\n",
    a->score_during_fit, a->fspt, a->node, a->discover, a->need_normalize,
    a->normalize_pass, a->compute_euristic_hyperparam, a->euristic_hyperparam,
    a->exponential_normalization,
    a->calibration_score, a->calibration_n_samples_p, a->calibration_volume_p,
    a->calibration_feat_length_p,
    a->volume_penalization, a->calibration_tau);
}

void save_score_args_file(FILE *fp, score_args *s, int *succ) {
    *succ &= fwrite(s, sizeof(score_args), 1, fp);
}

score_args *load_score_args_file(FILE *fp, int *succ) {
    score_args *s = malloc(sizeof(score_args));
    *succ &= fread(s, sizeof(score_args), 1, fp);
    return s;
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

#undef FLOAT_FORMAT__
#undef POINTER_FORMAT
#undef INTEGER_FORMAT
