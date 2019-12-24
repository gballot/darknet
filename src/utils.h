#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "darknet.h"
#include "list.h"

#define TIME(a) \
    do { \
        double start = what_time_is_it_now(); \
        a; \
        printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

#ifdef DEBUG
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif

#define debug_print(fmt, ...) \
    do { \
        if(DEBUG_TEST) \
            fprintf(stderr, "[%s:%d:%s()]  " fmt "\n", __FILE__, \
                    __LINE__, __func__, __VA_ARGS__); \
    } while (0)

#define debug_assert(bool) \
    do { \
        if (DEBUG_TEST) \
            assert(bool); \
    } while (0)

#define TIMEVAL_TO_TIMESPEC(tv, ts) \
    do { \
        (ts)->tv_sec = (tv)->tv_sec; \
        (ts)->tv_nsec = (tv)->tv_usec * 1000; \
    } while (0)

#define TIMESPEC_TO_TIMEVAL(tv, ts) \
    do { \
        (tv)->tv_sec = (ts)->tv_sec; \
        (tv)->tv_usec = (ts)->tv_nsec / 1000; \
    } while (0)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ABS(a) ((a) >= 0 ? (a) : -(a))

#define safe_div(a, b) ((b) != 0 ? (a) / (b) : 0)
#define safe_divf(a, b) ((b) != 0 ? (float) (a) / (b) : 0.f)
#define safe_divd(a, b) ((b) != 0 ? (double) (a) / (b) : 0.)

typedef struct polynome_t {
    long double a;
    long double b;
    long double c;
    int solved;
    long double delta;
    long double x1;
    long double x2;
} polynome_t;

extern double what_time_is_it_now();
extern void shuffle(void *arr, size_t n, size_t size);
extern void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
extern void free_ptrs(void **ptrs, int n);
extern int alphanum_to_int(char c);
extern char int_to_alphanum(int i);
extern int read_int(int fd);
extern void write_int(int fd, int n);
extern void read_all(int fd, char *buffer, size_t bytes);
extern void write_all(int fd, char *buffer, size_t bytes);
extern int read_all_fail(int fd, char *buffer, size_t bytes);
extern int write_all_fail(int fd, char *buffer, size_t bytes);
extern void find_replace(char *str, char *orig, char *rep, char *output);
extern void malloc_error();
extern void file_error(const char *s);
extern void strip(char *s);
extern void strip_char(char *s, char bad);
extern list *split_str(char *s, char delim);
extern char *fgetl(FILE *fp);
extern list *parse_csv_line(char *line);
extern char *copy_string(char *s);
extern int count_fields(char *line);
extern float *parse_fields(char *line, int n);
extern void translate_array(float *a, int n, float s);
extern float constrain(float min, float max, float a);
extern double constrain_double(double min, double max, double a);
extern long double constrain_long_double(long double min, long double max,
        long double a);
extern int constrain_int(int a, int min, int max);
extern float rand_scale(float s);
extern int rand_int(int min, int max);
extern void mean_arrays(float **a, int n, int els, float *avg);
extern float dist_array(float *a, float *b, int n, int sub);
extern float **one_hot_encode(float *a, int n, int k);
extern float sec(clock_t clocks);
extern void print_statistics(float *a, int n);
extern int int_index(int *a, int val, int n);
extern int max_index_double(double *a, int n);
extern int max_index_size_t(size_t *a, int n);
extern size_t *random_index_order_size_t(size_t min, size_t max);
extern char *itoa(int val, int base);
extern void qsort_float(size_t n, float *base);
extern void add_millis_to_timespec (struct timespec * ts, long msec);
extern void delay_until(struct timespec * deadline);
extern long elapsed_time();
extern struct timespec get_start_time();
extern void set_start_time();
extern int sum_array_int(int *a, int n);

/**
 * Implementation of the Quick Sort algorithm on bidimensional arrays of
 * size (n*size) according to the feature index. Ascending order.
 *
 * \param index The index of the feature to apply QSort. 0 <= index < size.
 * \param n The number of vectors in the array.
 * \param size The number of feature of each vectors.
 * \param base Output paramter. Pointer to the array of size (n*size).
 */
extern void qsort_float_on_index(size_t index, size_t n, size_t size,
        float *base);

/**
 * Gives the median of an array already sorted.
 *
 * \param a The array.
 * \param n_elem The number of elements in `a`.
 * \param size_elem The size of the elements in `a`.
 * \param accessor The function pointer to get the value of an element.
 */
extern double median(const void *a, size_t n_elem, size_t size_elem,
        double (*accessor) (const void *));

/**
 * Gives the first quartile of an array already sorted.
 *
 * \param a The array.
 * \param n_elem The number of elements in `a`.
 * \param size_elem The size of the elements in `a`.
 * \param accessor The function pointer to get the value of an element.
 */
extern double first_quartile(const void *a, size_t n_elem, size_t size_elem,
        double (*accessor) (const void *));

/**
 * Gives the third quartile of an array already sorted.
 *
 * \param a The array.
 * \param n_elem The number of elements in `a`.
 * \param size_elem The size of the elements in `a`.
 * \param accessor The function pointer to get the value of an element.
 */
extern double third_quartile(const void *a, size_t n_elem, size_t size_elem,
        double (*accessor) (const void *));

/**
 * Computes the roots and delta of a polynome.
 *
 * \param poly The polynome structure that will be completed.
 *             poly->a must be non null.
 */
extern void solve_polynome(polynome_t *poly);

/**
 * Computes the binomial coefficients
 * k among n. Uses a static pascal triangle and is probably
 * non thread safe.
 *
 * \param k The number of elements to choose among n.
 * \param n the total number of elements.
 * \return k among n.
 */
extern long binomial(int k, int n);

#endif

