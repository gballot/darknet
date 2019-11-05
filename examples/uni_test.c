#ifdef DEBUG
#include <stdlib.h>
#include <assert.h>

#include "list.h"
#include "utils.h"
#include "fspt.h"
#include "fspt_criterion.h"
#include "fspt_score.h"

static int eq_float_array(int n, const float *X, const float *Y) {
    for (int i = 0; i < n; ++i) {
        if (X[i] != Y[i])
            return 0;
    }
    return 1;
}

static int eq_nodes(fspt_node a, fspt_node b) {
    int eq = 1;
    eq &= (a.type == b.type);
    if (!eq) fprintf(stderr, "node types differ...\n");
    eq &= (a.n_features == b.n_features);
    if (!eq) fprintf(stderr, "node n_features differ...\n");
    if (!eq) return 0;
    eq &= (a.n_empty == b.n_empty);
    if (!eq) fprintf(stderr, "node n_empty differ...\n");
    eq &= (a.n_samples == b.n_samples);
    if (!eq) fprintf(stderr, "node n_samples differ...\n");
    if (!eq) return 0;
    eq &= eq_float_array(a.n_samples, a.samples, b.samples);
    if (!eq) fprintf(stderr, "node samples differ...\n");
    eq &= (a.split_feature == b.split_feature);
    if (!eq) fprintf(stderr, "node split_feature differ...\n");
    eq &= (a.split_value == b.split_value);
    if (!eq) fprintf(stderr, "node split_value differ...\n");
    if (a.right && b.right) 
        eq &= eq_nodes(*a.right, *b.right);
    if ((!a.right) != (!b.right)) eq = 0;
    if (!eq) fprintf(stderr, "node rigth differ...\n");
    if (a.left && b.left) 
        eq &= eq_nodes(*a.left, *b.left);
    if ((!a.left) != (!b.left)) eq = 0;
    if (!eq) fprintf(stderr, "node left differ...\n");
    eq &= (a.score == b.score);
    if (!eq) fprintf(stderr, "node score differ...\n");
    return eq;
}

static int eq_fspts(fspt_t a, fspt_t b) {
    int eq = 1;
    eq &= (a.n_features == b.n_features);
    if (!eq) fprintf(stderr, "fspt n_features differ...\n");
    if (!eq) return 0;
    eq &= eq_float_array(2*a.n_features, a.feature_limit, b.feature_limit);
    if (!eq) fprintf(stderr, "fspt feature_limit differ...\n");
    eq &= eq_float_array(a.n_features, a.feature_importance, b.feature_importance);
    if (!eq) fprintf(stderr, "fspt feature_importance differ...\n");
    eq &= (a.n_nodes == b.n_nodes);
    if (!eq) fprintf(stderr, "fspt n_nodes differ...\n");
    eq &= (a.n_samples == b.n_samples);
    if (!eq) fprintf(stderr, "fspt n_samples differ...\n");
    if (!eq) return 0;
    eq &= eq_float_array(a.n_samples, a.samples, b.samples);
    if (!eq) fprintf(stderr, "fspt samples differ...\n");
    eq &= eq_nodes(*a.root, *b.root);
    if (!eq) fprintf(stderr, "fspt root differ...\n");
    eq &= (a.criterion == b.criterion);
    if (!eq) fprintf(stderr, "fspt criterion differ...\n");
    eq &= (a.score == b.score);
    if (!eq) fprintf(stderr, "fspt score differ...\n");
    eq &= (a.count == b.count);
    if (!eq) fprintf(stderr, "fspt count differ...\n");
    return eq;
}


static void print_size_t_array(int lines, int col, size_t *X) {
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < col; ++j) {
            fprintf(stderr, " %zu  ", X[i*col +j]);
        }
        fprintf(stderr, "\n");
    }
}

static void print_array(int lines, int col, float *X) {
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < col; ++j) {
            fprintf(stderr, " %f  ", X[i*col +j]);
        }
        fprintf(stderr, "\n");
    }
}

void uni_test() {

    /***********************/
    /* Test qsort          */
    /***********************/

    float X[] = 
    {   
        0.1f , 0.5f ,
        -1.3f , 2.5f ,
        3.2f , 0.7f ,
        2.0f ,-1.5f , 
        5.9f  , 8.2f,
        2.7f , 1.7f,
        3.4f , 4.7f,
        1.2f ,-0.7f,
        -5.7f, -0.5f
    };

    float X_sorted_0[] =
    {   
        -5.7f, -0.5f,
        -1.3f , 2.5f ,
        0.1f , 0.5f ,
        1.2f ,-0.7f,
        2.0f ,-1.5f ,
        2.7f , 1.7f,
        3.2f , 0.7f,
        3.4f , 4.7f,
        5.9f  , 8.2f
    };

    float X_sorted_1[] =
    {   
        2.0f ,-1.5f ,
        1.2f ,-0.7f,
        -5.7f, -0.5f,
        0.1f , 0.5f ,
        3.2f , 0.7f ,
        2.7f , 1.7f,
        -1.3f , 2.5f,
        3.4f , 4.7f,
        5.9f  , 8.2f,
    };

    qsort_float_on_index(0, 9, 2, X);
    if(!eq_float_array(2*9, X, X_sorted_0)) {
        fprintf(stderr, "QSORT TESTS FAILD!\n");
        fprintf(stderr, "qsort_float_array(0, 9, 2, X) = \n");
        print_array(9, 2, X);
        fprintf(stderr, "instead of :\n");
        print_array(9, 2, X_sorted_0);
        error("UNI-TEST FAILD");
    }

    qsort_float_on_index(1, 9, 2, X);
    if(!eq_float_array(2*9, X, X_sorted_1)) {
        fprintf(stderr, "QSORT TESTS FAILD!\n");
        fprintf(stderr, "qsort_float_array(1, 9, 2, X) = \n");
        print_array(9, 2, X);
        fprintf(stderr, "instead of :\n");
        print_array(9, 2, X_sorted_1);
        error("UNI-TEST FAILD");
    }

    fprintf(stderr, "QSORT TESTS OK!\n");


    /***********************/
    /* Test fspt save/load */
    /***********************/
    
    float feat_lim[] = {0.f, 1.f, -1.f, 1.5f};
    float feat_lim2[] = {0.f, 1.f, -1.f, 1.5f};
    float samples[] = {
        0.1f, 1.f,
        0.2f, 0.5f,
        0.8f, -0.5f
    };
    int split_index = 0;
    float split_value = 0.5;
    float feat_imp[] = {1.f, 2.f};
    float feat_imp2[] = {1.f, 2.f};
    char *filename = "backup/uni_test_fspt.dat";
    int succ = 1;

    fspt_t *fspt = make_fspt(2, feat_lim, feat_imp, NULL, NULL);
    fspt->samples = samples;
    fspt->n_samples = 3;
    /* Builds left */
    fspt_node *left = calloc(1, sizeof(fspt_node));
    left->type = LEAF;
    left->n_features = fspt->n_features;
    left->depth = 2;
    left->samples = samples;
    left->n_samples = 2;
    /* Builds right */
    fspt_node *right = calloc(1, sizeof(fspt_node));
    right->type = LEAF;
    right->n_features = fspt->n_features;
    right->depth = 2;
    right->samples = samples + 2 * fspt->n_features;
    right->n_samples = fspt->n_samples - left->n_samples;
    /* Builds the root */
    fspt_node *root = calloc(1, sizeof(fspt_node));
    root->type = INNER;
    root->n_features = fspt->n_features;
    root->split_feature = split_index;
    root->split_value = split_value;
    root->depth = 1;
    root->left = left;
    root->right = right;
    root->samples = samples;
    root->n_samples = 3;
    /* Update fspt */
    fspt->n_nodes = 3;
    fspt->root = root;
    fspt->depth = 2;

    /* save */
    fspt_save(filename, *fspt, 1, &succ);
    print_fspt(fspt);

    if (!succ) {
        fprintf(stderr, "FSPT_SAVE FAILD\n");
        error("UNI-TEST FAILD");
    }
    fprintf(stderr, "sizeof(fspt_node) = %ld\n", sizeof(fspt_node));

    /* load */
    fspt_t *fspt_loaded = make_fspt(2, feat_lim2, feat_imp2, NULL, NULL);
    fspt_load(filename, fspt_loaded, 1, &succ);
    print_fspt(fspt_loaded);

    if (!succ) {
        fprintf(stderr, "FSPT_LOAD FAILD\n");
        error("UNI-TEST FAILD");
    }

    if (!eq_fspts(*fspt, *fspt_loaded)) {
        fprintf(stderr, "FSPT LOADED DIFFERENT FROM FSPT SAVED\n");
        error("UNI-TEST FAILD");
    }

    free_fspt(fspt);
    free_fspt(fspt_loaded);

    /***********************/
    /* Test fspt fit       */
    /***********************/
    
    float feat_lim_fit[] = {0.f, 1.f, -1.f, 0.f};
    float samples_fit[] = {
        0.2f, -0.8f,
        0.1f, -0.9f,
        0.2f, -0.9f,
        0.1f, -0.8f
    };
    float feat_imp_fit[] = {1.f, 1.f};
    int n_samples = 4;
    fspt_t *fspt_fitted = make_fspt(2, feat_lim_fit, feat_imp_fit,
            gini_criterion, euristic_score);
    criterion_args args = {0};
    args.fspt = fspt_fitted;
    args.max_tries_p = 1.f;
    args.max_features_p = 1.f;
    args.gini_gain_thresh = 0.1f;
    args.max_depth = 10;
    args.min_samples = 1;
    fspt_fit(n_samples, samples_fit, &args, fspt_fitted);
    print_fspt(fspt_fitted);

    free_fspt(fspt_fitted);

    /***********************/
    /* Test get_feat_limit */
    /***********************/

    float feat_lim_lim[] = {0.f, 1.f, 0.f, 1.f};
    float feat_imp_lim[] = {1.f, 1.f};
    /* alloc */
    fspt_node *root_lim = calloc(1,sizeof(fspt_node));
    fspt_node *left1 = calloc(1,sizeof(fspt_node));
    fspt_node *right1 = calloc(1,sizeof(fspt_node));
    fspt_node *left2 = calloc(1,sizeof(fspt_node));
    fspt_node *right2 = calloc(1,sizeof(fspt_node));
    fspt_node *left3 = calloc(1,sizeof(fspt_node));
    fspt_node *right3 = calloc(1,sizeof(fspt_node));
    /* fspt */
    fspt_t *fspt_lim = make_fspt(2, feat_lim_lim, feat_imp_lim,
            gini_criterion, euristic_score);
    /* rigth3 */
    right3->fspt = fspt_lim;
    right3->parent = right2;
    /* left3 */
    left3->fspt = fspt_lim;
    left3->parent = right2;
    /* right2 */
    right2->fspt = fspt_lim;
    right2->split_feature = 1;
    right2->split_value = 0.75f;
    right2->left = left3;
    right2->right = right3;
    right2->parent = left1;
    /* left2 */
    left2->fspt = fspt_lim;
    left2->parent = left1;
    /* right1 */
    right1->fspt = fspt_lim;
    right1->parent = root_lim;
    /* left1 */
    left1->fspt = fspt_lim;
    left1->split_feature = 1;
    left1->split_value = 0.5f;
    left1->left = left2;
    left1->right = right2;
    left1->parent = root_lim;
    /* root */
    root_lim->fspt = fspt_lim;
    root_lim->split_feature = 0;
    root_lim->split_value = 0.5f;
    root_lim->left = left1;
    root_lim->right = right1;

    fspt_lim->root = root_lim;

    float *feature_limit_l3 = get_feature_limit(left3);
    float real_feature_limit_l3[] = {
        0.f, 0.5f,
        0.5f, 0.75f
    };
    if (!eq_float_array(4, feature_limit_l3, real_feature_limit_l3)) {
        fprintf(stderr, "feature_limit_l3 =\n");
        print_array(2, 2, feature_limit_l3);
        fprintf(stderr, "instead of :\n");
        print_array(2, 2, real_feature_limit_l3);
        fprintf(stderr, "GET_FEATURE_LIMIT TEST FAILED.\n");
        error("UNI-TEST FAILED");
    }
    free_fspt(fspt_lim);
    free(feature_limit_l3);


    /***********************/
    /* Test hist           */
    /***********************/

    float X_hist[] = 
    {   
        0.1f , 0.5f ,
        -1.3f , 2.5f ,
        3.2f , 0.7f ,
        2.0f ,-1.5f , 
        5.9f  , 8.2f,
        2.7f , 1.7f,
        -10.f , 1.7f,
        0.1f , 0.0f ,
        3.4f , 4.7f,
        1.2f ,-0.7f,
        -5.7f, -0.5f
    };
    size_t n = 11;
    size_t step = 2;
    float lower = -10.f;
    size_t n_bins = 0;
    size_t *cdf = calloc(2*n, sizeof(size_t));
    float *bins = calloc(2*n, sizeof(float));

    qsort_float_on_index(0, n, step, X_hist);
    hist(n, step, X_hist, lower, &n_bins, cdf, bins);

    fprintf(stderr, "X = \n");
    print_array(n,step,X_hist);
    fprintf(stderr, "bins = \n");
    print_array(1, n_bins, bins);
    fprintf(stderr, "cdf = \n");
    print_size_t_array(1, n_bins, cdf);
    free(cdf);
    free(bins);

    fprintf(stderr, "ALL TESTS OK!\n");
}

#endif /* DEBUG */
