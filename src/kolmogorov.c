/**
 * Code from http://www.jstatsoft.org/v08/i18/paper?ev=pub_ext_btn_xdl
 * Evaluating Kolmogorov’s Distribution
 * George Marsaglia
 * The Florida State University
 * Wai Wan Tsang
 * Jingbo Wang
 * The University of Hong Kong
 */

#include "kolmogorov.h"

#include <math.h>
#include <stdlib.h>

static void mMultiply(double *A, double *B, double *C, int m) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double s = 0.;
            for (int k = 0; k < m; ++k) {
                s += A[i * m + k] * B[k * m + j];
            }
            C[i * m + j] = s;
        }
    }
}

static void mPower(double *A, int eA, double *V, int *eV, int m, int n) {
    if (n == 1) {
        for (int i = 0; i < m * m; ++i)
            V[i] = A[i];
        *eV = eA;
        return;
    }
    mPower(A, eA, V, eV, m, n / 2);
    double *B = (double *) malloc(m * m * sizeof(double));
    mMultiply(V, V, B, m);
    int eB = 2 * (*eV);
    if (n % 2 == 0) {
        for (int i = 0; i < m * m; ++i)
            V[i] = B[i];
        *eV=eB;
    } else {
        mMultiply(A, B, V, m);
        *eV = eA + eB;
    }
    if (V[(m / 2) * m + (m / 2)] > 1e140) {
        for (int i = 0; i < m * m; ++i)
            V[i] = V[i] * 1e-140;
        *eV += 140;
    }
    free(B);
}

double kolmogorov_p_value(int n, double d) {
    double s = d * d * n;
    if (s > 7.24 || (s > 3.76 && n > 99))
        return 1 - 2 * exp( - (2.000071 + .331 / sqrt(n) + 1.409 / n) * s);
    int k = (int) (n * d) + 1;
    int m = 2 * k - 1;
    double h = k - n * d;
    double *H = (double *) malloc(m * m * sizeof(double));
    double *Q = (double *) malloc(m * m * sizeof(double));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            if (i - j + 1 < 0)
                H[i * m + j] = 0;
            else
                H[i * m + j] = 1;
        }
    }
    for (int i = 0; i < m; ++i) {
        H[i * m] -= pow(h, i + 1);
        H[(m - 1) * m + i] -= pow(h, (m - i));
    }
    H[(m - 1) * m] += (2 * h - 1 > 0 ? pow(2 * h - 1, m) : 0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            if (i - j + 1 > 0) {
                for (int g = 1; g <= i - j + 1; ++g)
                    H[i * m + j] /= g;
            }
        }
    }
    int eH = 0;
    int eQ = 0;
    mPower(H, eH, Q, &eQ, m, n);
    s = Q[(k - 1) * m + k - 1];
    for (int i = 1; i <= n; ++i) {
        s = s * i / n;
        if (s < 1e-140) {
            s *= 1e140;
            eQ -= 140;
        }
    }
    s *= pow(10., eQ);
    free(H);
    free(Q);
    return s;
}
