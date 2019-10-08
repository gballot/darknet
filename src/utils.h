#ifndef UTILS_H
#define UTILS_H
#include "darknet.h"
#include "list.h"

#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

LIB_API void free_ptrs(void **ptrs, int n);
LIB_API void top_k(float *a, int n, int k, int *index);

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

extern double what_time_is_it_now();
extern int *read_map(char *filename);
extern void shuffle(void *arr, size_t n, size_t size);
extern void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
extern char *basecfg(char *cfgfile);
extern int alphanum_to_int(char c);
extern char int_to_alphanum(int i);
extern int read_int(int fd);
extern void write_int(int fd, int n);
extern void read_all(int fd, char *buffer, size_t bytes);
extern void write_all(int fd, char *buffer, size_t bytes);
extern int read_all_fail(int fd, char *buffer, size_t bytes);
extern int write_all_fail(int fd, char *buffer, size_t bytes);
extern LIB_API void find_replace(const char* str, char* orig, char* rep, char* output);
extern void replace_image_to_label(const char* input_path, char* output_path);
extern void error(const char *s);
extern void malloc_error();
extern void file_error(const char *s);
extern void strip(char *s);
extern void strip_args(char *s);
extern void strip_char(char *s, char bad);
extern list *split_str(char *s, char delim);
extern char *fgetl(FILE *fp);
extern list *parse_csv_line(char *line);
extern char *copy_string(char *s);
extern int count_fields(char *line);
extern float *parse_fields(char *line, int n);
extern void normalize_array(float *a, int n);
extern void scale_array(float *a, int n, float s);
extern void translate_array(float *a, int n, float s);
extern int max_index(float *a, int n);
extern int top_max_index(float *a, int n, int k);
extern float constrain(float min, float max, float a);
extern int constrain_int(int a, int min, int max);
extern float mse_array(float *a, int n);
extern float rand_normal();
extern size_t rand_size_t();
extern float rand_uniform(float min, float max);
extern float rand_scale(float s);
extern int rand_int(int min, int max);
extern float sum_array(float *a, int n);
extern float mean_array(float *a, int n);
extern void mean_arrays(float **a, int n, int els, float *avg);
extern float variance_array(float *a, int n);
extern float mag_array(float *a, int n);
extern float mag_array_skip(float *a, int n, int * indices_to_skip);
extern float dist_array(float *a, float *b, int n, int sub);
extern float **one_hot_encode(float *a, int n, int k);
extern float sec(clock_t clocks);
extern int find_int_arg(int argc, char **argv, char *arg, int def);
extern float find_float_arg(int argc, char **argv, char *arg, float def);
extern int find_arg(int argc, char* argv[], char *arg);
extern char *find_char_arg(int argc, char **argv, char *arg, char *def);
extern int sample_array(float *a, int n);
extern int sample_array_custom(float *a, int n);
extern void print_statistics(float *a, int n);
extern unsigned int random_gen();
extern float random_float();
extern float rand_uniform_strong(float min, float max);
extern float rand_precalc_random(float min, float max, float random_part);
extern double double_rand(void);
extern unsigned int uint_rand(unsigned int less_than);
extern int check_array_is_nan(float *arr, int size);
extern int check_array_is_inf(float *arr, int size);
extern int int_index(int *a, int val, int n);
extern int *random_index_order(int min, int max);
extern int max_int_index(int *a, int n);
extern char *itoa(int val, int base);

#ifdef __cplusplus
}
# endif

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


#endif
