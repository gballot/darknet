#ifdef DEBUG
#include <stdlib.h>
#include <assert.h>

#include "list.h"
#include "utils.h"
#include "fspt.h"
#include "fspt_criterion.h"

static float volume(int n_features, const float *feature_limit)
{
    float vol = 1;
    for (int i = 0; i < n_features; i+=2)
        vol *= feature_limit[i+1] - feature_limit[i];
    return vol;
}

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
    eq &= eq_float_array(2*a.n_features, a.feature_limit, b.feature_limit);
    if (!eq) fprintf(stderr, "node feature_limit differ...\n");
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
    eq &= (a.vol == b.vol);
    if (!eq) fprintf(stderr, "node vol differ...\n");
    eq &= (a.density == b.density);
    if (!eq) fprintf(stderr, "node density differ...\n");
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
    eq &= (a.vol == b.vol);
    if (!eq) fprintf(stderr, "fspt vol differ...\n");
    eq &= (a.max_depth == b.max_depth);
    if (!eq) fprintf(stderr, "fspt max_depth differ...\n");
    eq &= (a.min_samples == b.min_samples);
    if (!eq) fprintf(stderr, "fspt min_samples differ...\n");
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
    float feat_lim_left[] = {0.f, 0.5f, -1.f, 1.5f};
    float feat_lim_right[] = {0.5f, 1.f, -1.f, 1.5f};
    int split_index = 0;
    float split_value = 0.5;
    float feat_imp[] = {1.f, 2.f};
    int min_samp = 3;
    int max_depth = 2;
    char *filename = "backup/uni_test_fspt.dat";
    int succ = 1;

    fspt_t *fspt = make_fspt(2, feat_lim, feat_imp, NULL, NULL, min_samp, max_depth);
    /* Builds left */
    fspt_node *left = calloc(1, sizeof(fspt_node));
    left->type = LEAF;
    left->n_features = fspt->n_features;
    left->feature_limit = feat_lim_left;
    left->depth = 2;
    left->vol = volume(left->n_features, left->feature_limit);
    /* Builds right */
    fspt_node *right = calloc(1, sizeof(fspt_node));
    right->type = LEAF;
    right->n_features = fspt->n_features;
    right->feature_limit = feat_lim_right;
    right->depth = 2;
    right->vol = volume(right->n_features, right->feature_limit);
    /* Builds the root */
    fspt_node *root = calloc(1, sizeof(fspt_node));
    root->type = INNER;
    root->n_features = fspt->n_features;
    root->feature_limit = fspt->feature_limit;
    root->split_feature = split_index;
    root->split_value = split_value;
    root->depth = 1;
    root->left = left;
    root->right = right;
    root->vol = volume(root->n_features, root->feature_limit);
    /* Update fspt */
    fspt->n_nodes = 3;
    fspt->root = root;
    fspt->depth = 2;

    /* save */
    fspt_save(filename, *fspt, &succ);
    print_fspt(fspt);

    if (!succ) {
        fprintf(stderr, "FSPT_SAVE FAILD\n");
        error("UNI-TEST FAILD");
    }

    /* load */
    fspt_t *fspt_loaded = make_fspt(2, feat_lim, feat_imp, NULL, NULL, min_samp, max_depth);
    fspt_load(filename, fspt_loaded, &succ);

    if (!succ) {
        fprintf(stderr, "FSPT_LOAD FAILD\n");
        error("UNI-TEST FAILD");
    }

    if (!eq_fspts(*fspt, *fspt_loaded)) {
        fprintf(stderr, "FSPT LOADED DIFFERENT FROM FSPT SAVED\n");
        error("UNI-TEST FAILD");
    }


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

    fprintf(stderr, "ALL TESTS OK!\n");
}

#endif /* DEBUG */
