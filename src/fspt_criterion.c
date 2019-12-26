#include "fspt_criterion.h"

#include <stdlib.h>

#include "fspt.h"
#include "gini_utils.h"
#include "uniformity.h"
#include "utils.h"

#define FLOAT_FORMAT__ "%-16g"
#define POINTER_FORMAT "%-16p"
#define INTEGER_FORMAT "%-16d"
#define LONG_INTFORMAT "%-16ld"
#define CRITERION_ARGS_VERSION 6


int respect_min_lenght_p(int n_features, const float* fspt_lim,
        const float *node_lim, double min_length_p) {
    if (min_length_p == 0.) return 1;
    for (int i = 0; i < n_features; ++i) {
        float node_min = node_lim[2*i];
        float node_max = node_lim[2*i + 1];
        float fspt_min = fspt_lim[2*i];
        float fspt_max = fspt_lim[2*i + 1];
        double relative_length = (double) (node_max - node_min)
            / (fspt_max - fspt_min);
        if (relative_length < min_length_p) {
            return 0;
        }
    }
    return 1;
}

void determine_cause(int n, forbidden_split_cause *causes,
        criterion_args *args) {
    size_t tab[4] = {0};
    for (int i = 0; i < n; ++i) {
        forbidden_split_cause cause = causes[i];
        tab[0] += cause.count_min_volume_p_hit;
        tab[1] += cause.count_max_depth_hit;
        tab[2] += cause.count_min_samples_hit;
        tab[3] += cause.count_min_length_p_hit;
    }
    if (!tab[0] && !tab[1] && !tab[2] && !tab[3]) {
        args->node->cause = UNKNOWN_CAUSE;
        return;
    }
    int i = max_index_size_t(tab, 4);
    switch (i) {
        case 0:
            ++args->count_min_volume_p_hit;
            args->node->cause = MIN_VOLUME;
            break;
        case 1:
            ++args->count_max_depth_hit;
            args->node->cause = MAX_DEPTH;
            break;
        case 2:
            ++args->count_min_samples_hit;
            args->node->cause = MIN_SAMPLES;
            break;
        case 3:
            ++args->count_min_length_p_hit;
            args->node->cause = MIN_LENGTH;
            break;
        default: break;
    }
}

void print_fspt_criterion_args(FILE *stream, criterion_args *a, char *title) {
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
        fprintf(stream, "No criterion args.\n");
        return;
    }
    fprintf(stream, "\
┌──────────────────────────────────────────────┐\n\
│           FSPT CRITERION ARGUMENTS           │\n\
├──────────────────────────────────────────────┤\n\
│     Messages to change fitting behaviour     │\n\
├─────────────────────────────┬────────────────┤\n\
│                 merge_nodes │"INTEGER_FORMAT"│\n\
│          criterion_function │"INTEGER_FORMAT"│\n\
├─────────────────────────────┴────────────────┤\n\
│   Messages for all the criterion functions   │\n\
├─────────────────────────────┬────────────────┤\n\
│                        fspt │"POINTER_FORMAT"│\n\
│                        node │"POINTER_FORMAT"│\n\
│                   max_depth │"INTEGER_FORMAT"│\n\
│         count_max_depth_hit │"LONG_INTFORMAT"│\n\
│                 min_samples │"INTEGER_FORMAT"│\n\
│       count_min_samples_hit │"LONG_INTFORMAT"│\n\
│                min_volume_p │"FLOAT_FORMAT__"│\n\
│      count_min_volume_p_hit │"LONG_INTFORMAT"│\n\
│                min_length_p │"FLOAT_FORMAT__"│\n\
│      count_min_length_p_hit │"LONG_INTFORMAT"│\n\
│         count_max_count_hit │"LONG_INTFORMAT"│\n\
│         count_no_sample_hit │"LONG_INTFORMAT"│\n\
│        count_uniformity_hit │"LONG_INTFORMAT"│\n\
│                  best_index │"INTEGER_FORMAT"│\n\
│                  best_split │"FLOAT_FORMAT__"│\n\
│             forbidden_split │"INTEGER_FORMAT"│\n\
│             increment_count │"INTEGER_FORMAT"│\n\
│              end_of_fitting │"INTEGER_FORMAT"│\n\
├─────────────────────────────┴────────────────┤\n\
│         Messages for gini_criterion          │\n\
├─────────────────────────────┬────────────────┤\n\
│                 max_tries_p │"FLOAT_FORMAT__"│\n\
│              max_features_p │"FLOAT_FORMAT__"│\n\
│            gini_gain_thresh │"FLOAT_FORMAT__"│\n\
│max_consecutive_gain_violati │"INTEGER_FORMAT"│\n\
│                middle_split │"INTEGER_FORMAT"│\n\
│               multi_threads │"INTEGER_FORMAT"│\n\
│       uniformity_test_level │"INTEGER_FORMAT"│\n\
│                   unf_alpha │"FLOAT_FORMAT__"│\n\
└─────────────────────────────┴────────────────┘\n\n",
    a->merge_nodes, a->criterion_function,
    a->fspt, a->node,
    a->max_depth, a->count_max_depth_hit,
    a->min_samples, a->count_min_samples_hit,
    a->min_volume_p, a->count_min_volume_p_hit,
    a->min_length_p, a->count_min_length_p_hit,
    a->count_max_count_hit,
    a->count_no_sample_hit,
    a->count_uniformity_hit,
    a->best_index, a->best_split, a->forbidden_split,
    a->increment_count, a->end_of_fitting, a->max_tries_p, a->max_features_p,
    a->gini_gain_thresh, a->max_consecutive_gain_violations, a->middle_split,
    a->multi_threads,
    a->uniformity_test_level, a->unf_alpha);
}

int compare_criterion_args(const criterion_args *c1, const criterion_args *c2){
    CRITERION_FUNCTION f = c1->criterion_function;
    if (f != c2->criterion_function) return 0;
    int r = 1;
    r &= c1->merge_nodes == c2->merge_nodes;
    if (f == GINI || f == UNKNOWN_CRITERION_FUNC) {
        r &= (
            c1->max_depth == c2->max_depth
            && c1->min_samples == c2->min_samples
            && c1->min_volume_p == c2->min_volume_p
            && c1->min_length_p == c2->min_length_p
            && c1->max_tries_p == c2->max_tries_p
            && c1->max_features_p == c2->max_features_p
            && c1->gini_gain_thresh == c2->gini_gain_thresh
            && c1->max_consecutive_gain_violations
                == c2->max_consecutive_gain_violations
            && c1->middle_split == c2->middle_split
            && c1->uniformity_test_level == c2->uniformity_test_level
            && c1->unf_alpha == c2->unf_alpha
             );
    }
    return r;
}

void save_criterion_args_file(FILE *fp, criterion_args *c, int *succ) {
    int contains_args = 0;
    int version = CRITERION_ARGS_VERSION;
    if (c) {
        contains_args = 1;
        size_t size = sizeof(criterion_args);
        *succ &= fwrite(&contains_args, sizeof(int), 1, fp);
        *succ &= fwrite(&version, sizeof(int), 1, fp);
        *succ &= fwrite(&size, sizeof(size_t), 1, fp);
        *succ &= fwrite(c, sizeof(criterion_args), 1, fp);
    } else {
        *succ &= fwrite(&contains_args, sizeof(int), 1, fp);
    }
}

criterion_args *load_criterion_args_file(FILE *fp, int *succ) {
    criterion_args *c = NULL;
    int contains_args = 0;
    int version = 0;
    size_t size = 0;
    *succ &= fread(&contains_args, sizeof(int), 1, fp);
    if (contains_args == 1) {
        *succ &= fread(&version, sizeof(int), 1, fp);
        *succ &= fread(&size, sizeof(size_t), 1, fp);
        if (version == CRITERION_ARGS_VERSION
                && size == sizeof(criterion_args)) {
            c = malloc(sizeof(criterion_args));
            *succ &= fread(c, sizeof(criterion_args), 1, fp);
        } else {
            fseek(fp, size, SEEK_CUR);
            fprintf(stderr, "Wrong criterion args version (%d) or size \
(saved size = %ld and sizeof(criterion_args) = %ld).\n",
                    version, size, sizeof(criterion_args));
        }
    } else if (contains_args != 0) {
        fprintf(stderr, "ERROR : in load_criterion_args_file - contains_args = %d.\n",
                contains_args);
    }
    return c;
}

CRITERION_FUNCTION string_to_criterion_function_number(char *s) {
    if (strcmp(s, "gini") == 0) {
        return GINI;
    } else {
        return UNKNOWN_CRITERION_FUNC;
    }
}

criterion_func string_to_fspt_criterion(char *s) {
    if (strcmp(s, "gini") == 0) {
        return gini_criterion;
    } else {
        return NULL;
    }
}

#undef FLOAT_FORMAT__
#undef POINTER_FORMAT
#undef INTEGER_FORMAT
#undef CRITERION_ARGS_VERSION
#undef LONG_INTFORMAT

