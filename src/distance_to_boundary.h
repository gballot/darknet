#ifndef DISTANCE_TO_BOUNDARY_H
#define DISTANCE_TO_BOUNDARY_H

/**
 * Returns the p-value of the K-S test to reject the hypothesis of
 * uniformity. When a p-value is less than or equal to the significance level,
 * you reject the null hypothesis.
 * \param d The number of dimensions.
 * \param n The number of samples.
 * \param X The samples, the first d elements is the first vector.
 * \return The p-value of the distance to bound test.
 */
extern double dist_to_bound_test(int d, int n, const float *X,
        const float *lim);

#endif /* not DISTANCE_TO_BOUNDARY_H */
