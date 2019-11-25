#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
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
extern size_t *random_index_order_size_t(size_t min, size_t max);
extern char *itoa(int val, int base);

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


#endif

