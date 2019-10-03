#include <stdlib.h>
#include <assert.h>

#include "utils.h"

static int eq_float_array(int n, const float *X, const float *Y) {
    for (int i = 0; i < n; ++i) {
        if (X[i] != Y[i])
            return 0;
    }
    return 1;
}

static print_array(int lines, int col, float *X) {
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < col; ++j) {
            fprintf(stderr, " %f  ", X[i*col +j]);
        }
        fprintf(stderr, "\n");
    }
}

void uni_test() {
    float X[] = 
    {   
        0.1f , 0.5f ,
        -1.3f , 2.5f ,
        3.2f , 0.7f ,
        2.0f ,-1.5f , 
        5.9f  , 8.2f,
        -5.7f, -0.5f
    };

    float X_sorted_0[] =
    {   
        -5.7f, -0.5f,
        -1.3f , 2.5f ,
        0.1f , 0.5f ,
        2.0f ,-1.5f ,
        3.2f , 0.7f,
        5.9f  , 8.2f
    };

    float X_sorted_1[] =
    {   
        2.0f ,-1.5f ,
        -5.7f, -0.5f,
        0.1f , 0.5f ,
        3.2f , 0.7f ,
        -1.3f , 2.5f,
        5.9f  , 8.2f,
    };

    qsort_float_on_index(0, 6, 2, X);
    if(!eq_float_array(2*6, X, X_sorted_0)) {
        fprintf(stderr, "QSORT TESTS FAILD!\n");
        fprintf(stderr, "qsort_float_array(0, 6, 2, X) = \n");
        print_array(6, 2, X);
        fprintf(stderr, "instead of :\n");
        print_array(6, 2, X_sorted_0);
        error("UNI-TEST FAILD");
    }

    qsort_float_on_index(1, 6, 2, X);
    if(!eq_float_array(2*6, X, X_sorted_1)) {
        fprintf(stderr, "QSORT TESTS FAILD!\n");
        fprintf(stderr, "qsort_float_array(1, 6, 2, X) = \n");
        print_array(6, 2, X);
        print_array(6, 2, X_sorted_1);
        fprintf(stderr, "instead of :\n");
        error("UNI-TEST FAILD");
    }

    fprintf(stderr, "QSORT TESTS OK!\n");

    fprintf(stderr, "ALL TESTS OK!\n");
}
