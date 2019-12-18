#ifndef KOLMOGOROV_H
#define KOLMOGOROV_H

/**
 * Computes the p-value of the Kolmogorov-Smirnov test.
 * That is to say :
 *   P(D_n < d) with D_n the K-S goodness of fit.
 *
 * \param n The number of samples.
 * \param d The critical value.
 * \return The level of significance of a K-S test of n samples with output d.
 */
extern double kolmogorov_p_value(int n, double d);

#endif /* not KOLMOGOROV_H */
