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
#define LONGINT_FORMAT "%-16ld"
#define SCORE_ARGS_VERSION 4

static double auto_normalize(density_normalize_args a, double raw_score) {
    if (!a.verification_passed) return 0.;
    double score = 1. - exp(- a.tau * raw_score);
    return score;
}

static void compute_norm_args(score_args *s_args) {
    if (!s_args->fspt->n_samples || !s_args->n_leaves) return;
    double samples_p = s_args->samples_p;
    size_t samples_break = s_args->fspt->n_samples * samples_p;
    size_t samples_count = 0;
    size_t uniform_leaves = 0;
    double volume_p_count = 0.;
    size_t i_break = 0;
    for (i_break = 0; i_break < s_args->n_leaves; ++i_break) {
        score_vol_n svn = s_args->score_vol_n_array[i_break];
        samples_count += svn.n_samples;
        volume_p_count += svn.volume_p;
        if (svn.cause == MERGE
                || svn.cause == MAX_COUNT
                || svn.cause == UNIFORMITY)
            ++uniform_leaves;
        if (samples_count >= samples_break) break;
    }
    s_args->norm_args.verification_passed = 1;
    if (s_args->verify_density_thresh) {
        double density_count = (double) samples_count / volume_p_count
            / s_args->fspt->n_samples;
        debug_print("density_count density_thresh = %g, %g.\n", density_count,
                s_args->verify_density_thresh);
        if (density_count < s_args->verify_density_thresh)
            s_args->norm_args.verification_passed = 0;
    }
    if (s_args->verify_n_nodes_p_thresh) {
        size_t n_nodes_thresh =
            ceil(s_args->n_leaves * s_args->verify_n_nodes_p_thresh);
        debug_print("i_break n_nodes_thresh = %ld, %ld.\n", i_break,
                n_nodes_thresh);
        if (i_break > n_nodes_thresh)
            s_args->norm_args.verification_passed = 0;
    }
    if (s_args->verify_n_uniform_p_thresh) {
        size_t n_uniform_thresh =
            floor((i_break + 1) * s_args->verify_n_uniform_p_thresh);
        debug_print("uniform_leaves n_uniform_thresh = %ld, %ld.\n",
                uniform_leaves, n_uniform_thresh);
        if (uniform_leaves < n_uniform_thresh)
            s_args->norm_args.verification_passed = 0;
    }
    double raw_score = s_args->score_vol_n_array[i_break].score;
    debug_print("for i_break : score, volume_p, n_samples = %f, %f, %ld",
            raw_score,s_args->score_vol_n_array[i_break].volume_p,
            s_args->score_vol_n_array[i_break].n_samples);
    debug_print("i_break, raw score = %ld, %f", i_break, raw_score);
    assert(raw_score);
    s_args->norm_args.tau =
        - log(1. - s_args->auto_calibration_score) / raw_score;
    debug_print("tau %f", s_args->norm_args.tau);
}

double auto_normalized_density_score(score_args *args) {
    fspt_node *node = args->node;
    fspt_t *fspt = args->fspt;
    if (args->discover) {
        args->discover = 0;
        args->need_normalize = 1;
        args->compute_norm_args = 1;
        args->score_during_fit = 0;
        return 0.;
    }
    if (args->normalize_pass) {
        if (args->compute_norm_args) {
            compute_norm_args(args);
            args->compute_norm_args = 0;
        }
        return auto_normalize(args->norm_args, node->score);
    } else {
        if (fspt->n_samples == 0) return 0.;
        double score = ((double) node->n_samples / fspt->n_samples)
            * (fspt->volume / node->volume);
        return score;
    }
}

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
        return 0.;
    }
    if (fspt->n_samples == 0) return 0.;
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
    if (!a) {
        fprintf(stream, "No score args.\n");
        return;
    }
    fprintf(stream, "\
┌──────────────────────────────────────────────┐\n\
│             FSPT SCORE ARGUMENTS             │\n\
├──────────────────────────────────────────────┤\n\
│     Messages to change fitting behaviour     │\n\
├─────────────────────────────┬────────────────┤\n\
│            score_during_fit │"INTEGER_FORMAT"│\n\
│              score_function │"INTEGER_FORMAT"│\n\
├─────────────────────────────┴────────────────┤\n\
│     Message for all the score functions      │\n\
├─────────────────────────────┬────────────────┤\n\
│                        fspt │"POINTER_FORMAT"│\n\
│                        node │"POINTER_FORMAT"│\n\
│                    discover │"INTEGER_FORMAT"│\n\
│              need_normalize │"INTEGER_FORMAT"│\n\
│              normalize_pass │"INTEGER_FORMAT"│\n\
│                    n_leaves │"LONGINT_FORMAT"│\n\
│           score_vol_n_array │"POINTER_FORMAT"│\n\
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
├─────────────────────────────┴────────────────┤\n\
│         Messages for auto_density_score      │\n\
├─────────────────────────────┬────────────────┤\n\
│           compute_norm_args │"INTEGER_FORMAT"│\n\
│                   samples_p │"FLOAT_FORMAT__"│\n\
│       verify_density_thresh │"FLOAT_FORMAT__"│\n\
│     verify_n_nodes_p_thresh │"FLOAT_FORMAT__"│\n\
│   verify_n_uniform_p_thresh │"FLOAT_FORMAT__"│\n\
│      auto_calibration_score │"FLOAT_FORMAT__"│\n\
│               norm_args.tau │"FLOAT_FORMAT__"│\n\
│norm_args.verification_passed│"INTEGER_FORMAT"│\n\
└─────────────────────────────┴────────────────┘\n\n",
    a->score_during_fit, a->score_function,
    a->fspt, a->node, a->discover, a->need_normalize,
    a->normalize_pass, a->n_leaves,
    a->score_vol_n_array,
    a->compute_euristic_hyperparam, a->euristic_hyperparam,
    a->exponential_normalization,
    a->calibration_score, a->calibration_n_samples_p, a->calibration_volume_p,
    a->calibration_feat_length_p,
    a->volume_penalization, a->calibration_tau,
    a->compute_norm_args, a->samples_p, a->verify_density_thresh,
    a->verify_n_nodes_p_thresh, a->verify_n_uniform_p_thresh,
    a->auto_calibration_score,
    a->norm_args.tau, a->norm_args.verification_passed);
}

int compare_score_args(const score_args *s1, const score_args *s2) {
    if (!s1 || !s2) return 0;
    SCORE_FUNCTION f = s1->score_function;
    if (f != s2->score_function) return 0;
    int r = 1;
    if (f == EURISTIC || f == UNKNOWN_SCORE_FUNC) {
    }
    if (f == DENSITY || f == UNKNOWN_SCORE_FUNC) {
        r &= (
            s1->exponential_normalization == s2->exponential_normalization
            && s1->calibration_score == s2->calibration_score
            && s1->calibration_n_samples_p == s2->calibration_n_samples_p
            && (s1->calibration_volume_p == s2->calibration_volume_p
                || !s1->calibration_volume_p || !s2->calibration_volume_p)
            && s1->calibration_feat_length_p == s2->calibration_feat_length_p
            && s1->volume_penalization == s2->volume_penalization
            );
    }
    if (f == AUTO_DENSITY || f == UNKNOWN_SCORE_FUNC) {
        r &= (
            s1->samples_p == s2->samples_p
            && s1->verify_density_thresh == s2->verify_density_thresh
            && s1->verify_n_nodes_p_thresh == s2->verify_n_nodes_p_thresh
            && s1->verify_n_uniform_p_thresh == s2->verify_n_uniform_p_thresh
            && s1->auto_calibration_score == s2->auto_calibration_score
            );
    }
    return r;
}

void save_score_args_file(FILE *fp, score_args *s, int *succ) {
    int contains_args = 0;
    int version = SCORE_ARGS_VERSION;
    if (s) {
        contains_args = 1;
        size_t size = sizeof(score_args);
        *succ &= fwrite(&contains_args, sizeof(int), 1, fp);
        *succ &= fwrite(&version, sizeof(int), 1, fp);
        *succ &= fwrite(&size, sizeof(size_t), 1, fp);
        *succ &= fwrite(s, sizeof(score_args), 1, fp);
    } else {
        *succ &= fwrite(&contains_args, sizeof(int), 1, fp);
    }
}

score_args *load_score_args_file(FILE *fp, int *succ) {
    score_args *s = NULL;
    int contains_args = 0;
    int version = 0;
    size_t size = 0;
    *succ &= fread(&contains_args, sizeof(int), 1, fp);
    if (contains_args == 1) {
        *succ &= fread(&version, sizeof(int), 1, fp);
        *succ &= fread(&size, sizeof(size_t), 1, fp);
        if (version == SCORE_ARGS_VERSION
                && size == sizeof(score_args)) {
            s = malloc(sizeof(score_args));
            *succ &= fread(s, sizeof(score_args), 1, fp);
        } else {
            fseek(fp, size, SEEK_CUR);
            fprintf(stderr, "Wrong score args version (%d) or size\
(saved size = %ld and sizeof(score_args) = %ld).\n",
                    version, size, sizeof(score_args));
        }
    } else if (contains_args != 0) {
        fprintf(stderr, "ERROR : in load_score_args_file - contains_args = %d.\n",
                contains_args);
    }
    return s;
}

SCORE_FUNCTION string_to_score_function_number(char *s) {
    if (strcmp(s, "euristic") == 0) {
        return EURISTIC;
    }  else if (strcmp(s, "density") == 0) {
        return DENSITY;
    }  else if (strcmp(s, "auto_density") == 0) {
        return AUTO_DENSITY;
    } else {
        return UNKNOWN_SCORE_FUNC;
    }
}

score_func string_to_fspt_score(char *s) {
    if (strcmp(s, "euristic") == 0) {
        return euristic_score;
    }  else if (strcmp(s, "density") == 0) {
        return density_score;
    }  else if (strcmp(s, "auto_density") == 0) {
        return auto_normalized_density_score;
    } else {
        return NULL;
    }
}

#undef FLOAT_FORMAT__
#undef POINTER_FORMAT
#undef INTEGER_FORMAT
#undef LONGINT_FORMAT
#undef SCORE_ARGS_VERSION

