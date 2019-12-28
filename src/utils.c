#include "utils.h"

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "prng.h"

#define RAND() prng_get_int()
#define L_RAND_MAX INT_MAX

typedef struct pascal_t {
    int n_max;
    long **t;
    pthread_mutex_t m;
} pascal_t;

static volatile pascal_t pascal = {0, 0, PTHREAD_MUTEX_INITIALIZER};

// Start time as a timespec
struct timespec start_time;

// Add msec milliseconds to timespec ts (seconds, nanoseconds)
void add_millis_to_timespec (struct timespec * ts, long msec) {
  long nsec = (msec % (long) 1E3) * 1E6;
  long  sec = msec / 1E3;
  ts->tv_nsec = ts->tv_nsec + nsec;
  if (1E9 <= ts->tv_nsec) {
    ts->tv_nsec = ts->tv_nsec - 1E9;
    ts->tv_sec++;
  }
  ts->tv_sec = ts->tv_sec + sec;
}

// Delay until an absolute time. Translate the absolute time into a
// relative one and use nanosleep. This is incorrect (we fix that).
void delay_until(struct timespec * deadline) {
  struct timeval  tv_now;
  struct timespec ts_now;
  struct timespec ts_sleep;

  gettimeofday(&tv_now, NULL);
  TIMEVAL_TO_TIMESPEC(&tv_now, &ts_now);
  ts_sleep.tv_nsec = deadline->tv_nsec - ts_now.tv_nsec;
  ts_sleep.tv_sec = deadline->tv_sec - ts_now.tv_sec;
  if (ts_sleep.tv_nsec < 0) {
    ts_sleep.tv_nsec = 1E9 + ts_sleep.tv_nsec;
    ts_sleep.tv_sec--;
  }
  if (ts_sleep.tv_sec < 0) return;
  
  nanosleep (&ts_sleep, &ts_now);
}

// Compute time elapsed from start time
long elapsed_time() {
  struct timeval  tv_now;
  struct timespec ts_now;

  gettimeofday(&tv_now, NULL);
  TIMEVAL_TO_TIMESPEC(&tv_now, &ts_now);
  
  ts_now.tv_nsec = ts_now.tv_nsec - start_time.tv_nsec;
  ts_now.tv_sec = ts_now.tv_sec - start_time.tv_sec;
  if (ts_now.tv_nsec < 0) {
    ts_now.tv_sec = ts_now.tv_sec - 1;
    ts_now.tv_nsec = ts_now.tv_nsec + 1E9;
  }
  return (ts_now.tv_sec * 1E3) + (ts_now.tv_nsec / 1E6);
}

// Return the start time
struct timespec get_start_time() {
  return start_time;
}

// Store current time as the start time
void set_start_time() {
  struct timeval  tv_start_time; // start time as a timeval
  
  gettimeofday(&tv_start_time, NULL);
  TIMEVAL_TO_TIMESPEC(&tv_start_time, &start_time);
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int *read_intlist(char *gpu_list, int *ngpus, int d)
{
    int *gpus = 0;
    if(gpu_list){
        int len = strlen(gpu_list);
        *ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++*ngpus;
        }
        gpus = calloc(*ngpus, sizeof(int));
        for(i = 0; i < *ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpus = calloc(1, sizeof(float));
        *gpus = d;
        *ngpus = 1;
    }
    return gpus;
}

int *read_map(char *filename)
{
    int n = 0;
    int *map = 0;
    char *str;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    while((str=fgetl(file))){
        ++n;
        map = realloc(map, n*sizeof(int));
        map[n-1] = atoi(str);
    }
    return map;
}

void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections)
{
    size_t i;
    for(i = 0; i < sections; ++i){
        size_t start = n*i/sections;
        size_t end = n*(i+1)/sections;
        size_t num = end-start;
        shuffle(arr+(start*size), num, size);
    }
}

void shuffle(void *arr, size_t n, size_t size)
{
    size_t i;
    void *swp = calloc(1, size);
    for(i = 0; i < n-1; ++i){
        size_t j = i + RAND()/(L_RAND_MAX / (n-i)+1);
        memcpy(swp,          arr+(j*size), size);
        memcpy(arr+(j*size), arr+(i*size), size);
        memcpy(arr+(i*size), swp,          size);
    }
}

size_t *random_index_order_size_t(size_t min, size_t max)
{
    size_t *inds = calloc(max-min, sizeof(size_t));
    size_t i;
    for(i = min; i < max; ++i){
        inds[i] = i;
    }
    if (max == 0) return inds;
    for(i = min; i < max-1; ++i){
        int swap = inds[i];
        int index = i + RAND()%(max-i);
        inds[i] = inds[index];
        inds[index] = swap;
    }
    return inds;
}

int *random_index_order(int min, int max)
{
    int *inds = calloc(max-min, sizeof(int));
    int i;
    for(i = min; i < max; ++i){
        inds[i] = i;
    }
    for(i = min; i < max-1; ++i){
        int swap = inds[i];
        int index = i + RAND()%(max-i);
        inds[i] = inds[index];
        inds[index] = swap;
    }
    return inds;
}

void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i) {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

float find_float_arg(int argc, char **argv, char *arg, float def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atof(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}


char *basecfg(char *cfgfile)
{
    char *c = cfgfile;
    char *next;
    while((next = strchr(c, '/')))
    {
        c = next+1;
    }
    c = copy_string(c);
    next = strchr(c, '.');
    if (next) *next = 0;
    return c;
}

int alphanum_to_int(char c)
{
    return (c < 58) ? c - 48 : c-87;
}
char int_to_alphanum(int i)
{
    if (i == 36) return '.';
    return (i < 10) ? i + 48 : i + 87;
}

void pm(int M, int N, float *A)
{
    int i,j;
    for(i =0 ; i < M; ++i){
        printf("%d ", i+1);
        for(j = 0; j < N; ++j){
            printf("%2.4f, ", A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void find_replace(char *str, char *orig, char *rep, char *output)
{
    char buffer[4096] = {0};
    char *p;

    sprintf(buffer, "%s", str);
    if(!(p = strstr(buffer, orig))){  // Is 'orig' even in 'str'?
        if (output != str) sprintf(output, "%s", str);
        return;
    }

    *p = '\0';

    sprintf(output, "%s%s%s", buffer, rep, p+strlen(orig));
}

float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}

void top_k(float *a, int n, int k, int *index)
{
    int i,j;
    for(j = 0; j < k; ++j) index[j] = -1;
    for(i = 0; i < n; ++i){
        int curr = i;
        for(j = 0; j < k; ++j){
            if((index[j] < 0) || a[curr] > a[index[j]]){
                int swap = curr;
                curr = index[j];
                index[j] = swap;
            }
        }
    }
}

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

unsigned char *read_file(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    size_t size;

    fseek(fp, 0, SEEK_END); 
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET); 

    unsigned char *text = calloc(size+1, sizeof(char));
    fread(text, 1, size, fp);
    fclose(fp);
    return text;
}

void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

void file_error(const char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

list *split_str(char *s, char delim)
{
    size_t i;
    size_t len = strlen(s);
    list *l = make_list();
    list_insert(l, s);
    for(i = 0; i < len; ++i){
        if(s[i] == delim){
            s[i] = '\0';
            list_insert(l, &(s[i+1]));
        }
    }
    return l;
}

void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

void strip_char(char *s, char bad)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==bad) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

void free_ptrs(void **ptrs, int n)
{
    int i;
    for(i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = realloc(line, size*sizeof(char));
            if(!line) {
                printf("%ld\n", size);
                malloc_error();
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}

int read_int(int fd)
{
    int n = 0;
    int next = read(fd, &n, sizeof(int));
    if(next <= 0) return -1;
    return n;
}

void write_int(int fd, int n)
{
    int next = write(fd, &n, sizeof(int));
    if(next <= 0) error("read failed");
}

int read_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}

int write_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}

void read_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) error("read failed");
        n += next;
    }
}

void write_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) error("write failed");
        n += next;
    }
}


char *copy_string(char *s)
{
    char *copy = malloc(strlen(s)+1);
    strncpy(copy, s, strlen(s)+1);
    return copy;
}

list *parse_csv_line(char *line)
{
    list *l = make_list();
    char *c, *p;
    int in = 0;
    for(c = line, p = line; *c != '\0'; ++c){
        if(*c == '"') in = !in;
        else if(*c == ',' && !in){
            *c = '\0';
            list_insert(l, copy_string(p));
            p = c+1;
        }
    }
    list_insert(l, copy_string(p));
    return l;
}

int count_fields(char *line)
{
    int count = 0;
    int done = 0;
    char *c;
    for(c = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done) ++count;
    }
    return count;
}

float *parse_fields(char *line, int n)
{
    float *field = calloc(n, sizeof(float));
    char *c, *p, *end;
    int count = 0;
    int done = 0;
    for(c = line, p = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done){
            *c = '\0';
            field[count] = strtod(p, &end);
            if(p == c) field[count] = nan("");
            if(end != c && (end != c-1 || *end != '\r')) field[count] = nan(""); //DOS file formats!
            p = c+1;
            ++count;
        }
    }
    return field;
}

int sum_array_int(int *a, int n)
{
    int i;
    int sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}

float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}

float mean_array(float *a, int n)
{
    return sum_array(a,n)/n;
}

void mean_arrays(float **a, int n, int els, float *avg)
{
    int i;
    int j;
    memset(avg, 0, els*sizeof(float));
    for(j = 0; j < n; ++j){
        for(i = 0; i < els; ++i){
            avg[i] += a[j][i];
        }
    }
    for(i = 0; i < els; ++i){
        avg[i] /= n;
    }
}

void print_statistics(float *a, int n)
{
    float m = mean_array(a, n);
    float v = variance_array(a, n);
    printf("MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array(a, n), m, v);
}

float variance_array(float *a, int n)
{
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for(i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i]-mean);
    float variance = sum/n;
    return variance;
}

int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

float constrain(float min, float max, float a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

double constrain_double(double min, double max, double a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

long double constrain_long_double(long double min, long double max,
        long double a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

float dist_array(float *a, float *b, int n, int sub)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; i += sub) sum += pow(a[i]-b[i], 2);
    return sqrt(sum);
}

float mse_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i]*a[i];
    return sqrt(sum/n);
}

void normalize_array(float *a, int n)
{
    int i;
    float mu = mean_array(a,n);
    float sigma = sqrt(variance_array(a,n));
    for(i = 0; i < n; ++i){
        a[i] = (a[i] - mu)/sigma;
    }
    mu = mean_array(a,n);
    sigma = sqrt(variance_array(a,n));
}

void translate_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] += s;
    }
}

float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        sum += a[i]*a[i];   
    }
    return sqrt(sum);
}

void scale_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] *= s;
    }
}

int sample_array(float *a, int n)
{
    float sum = sum_array(a, n);
    scale_array(a, n, 1./sum);
    float r = rand_uniform(0, 1);
    int i;
    for(i = 0; i < n; ++i){
        r = r - a[i];
        if (r <= 0) return i;
    }
    return n-1;
}

int max_int_index(int *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    int max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

int max_index_double(double *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    double max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

int max_index_size_t(size_t *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    size_t max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

int int_index(int *a, int val, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        if(a[i] == val) return i;
    }
    return -1;
}

int rand_int(int min, int max)
{
    if (max < min){
        int s = min;
        min = max;
        max = s;
    }
    int r = (RAND()%(max - min + 1)) + min;
    return r;
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
    static int haveSpare = 0;
    static double rand1, rand2;

    if(haveSpare)
    {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = RAND() / ((double) L_RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (RAND() / ((double) L_RAND_MAX)) * TWO_PI;

    return sqrt(rand1) * cos(rand2);
}

/*
   float rand_normal()
   {
   int n = 12;
   int i;
   float sum= 0;
   for(i = 0; i < n; ++i) sum += (float)RAND()/L_RAND_MAX;
   return sum-n/2.;
   }
 */

size_t rand_size_t()
{
    return  ((size_t)(RAND()&0xff) << 56) | 
        ((size_t)(RAND()&0xff) << 48) |
        ((size_t)(RAND()&0xff) << 40) |
        ((size_t)(RAND()&0xff) << 32) |
        ((size_t)(RAND()&0xff) << 24) |
        ((size_t)(RAND()&0xff) << 16) |
        ((size_t)(RAND()&0xff) << 8) |
        ((size_t)(RAND()&0xff) << 0);
}

float rand_uniform(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)RAND()/L_RAND_MAX * (max - min)) + min;
}

float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if(RAND()%2) return scale;
    return 1./scale;
}

float **one_hot_encode(float *a, int n, int k)
{
    int i;
    float **t = calloc(n, sizeof(float*));
    for(i = 0; i < n; ++i){
        t[i] = calloc(k, sizeof(float));
        int index = (int)a[i];
        t[i][index] = 1;
    }
    return t;
}

char *itoa(int val, int base)
{
    static char buf[32] = {0};
    int i = 30;
    int neg = 0;
    if(val < 0) {
        neg = 1;
        val = -val;
    }
    if(val == 0) {
        buf[i] = '0';
        return &buf[i];
    }
    for(; val && i ; --i, val /= base)
        buf[i] = "0123456789abcdef"[val % base];
    if(neg) {
        buf[i] = '-';
        return &buf[i];
    }
    return &buf[i+1];
}


/**
 * Helper funciton for the Quick Sort algorithm.
 * Puts smaller values before a choosen pivot and greater values after.
 *
 * \param index The index of the feature to apply QSort. 0 <= index < size.
 * \param n The number of vectors in the array.
 * \param size The number of feature of each vectors.
 * \param base Output paramter. Pointer to the array of size (n*size).
 * \return The size of the left subarray that has only values inferior
 *         or equal to the pivot value. The rest of the array has values
 *         supperior or equal to the pivot.
 */
static int partition(size_t index, size_t n, size_t size, float *base) {
    float *pivot = malloc(size * sizeof(float));
    for (size_t i = 0; i < size; ++i) {
        pivot[i] = base[(n-1)/2 * size + i];
    }
    int i = -1;
    int j = n;
    while(1) {
        do { ++i; } while (base[i*size + index] < pivot[index]);
        do { --j; } while (base[j*size + index] > pivot[index]);
        if (i >= j) {
            free(pivot);
            return j + 1;
        }
        /* Swap i and j. */
        for (size_t k = 0; k < size; ++k) {
            float swap = base[i*size + k];
            base[i*size + k] = base[j*size + k];
            base[j*size + k] = swap;
        }
    }
}

void qsort_float_on_index(size_t index, size_t n, size_t size,
                                 float *base) {
    assert(n < 10000000);
    if (n == 2) {
        if (base[index] > base[size + index]) {
            for (size_t i = 0; i < size; ++i) {
                float swap = base[i];
                base[i] = base[size + i];
                base[size + i] = swap;
            }
        }
    } else if (n > 2) {
        int p = partition(index, n, size, base);
        qsort_float_on_index(index, p, size, base);
        qsort_float_on_index(index, n - p, size, base + p * size);
    }
}

static int cmp_float(const void *fp1, const void *fp2) {
    float f1 = *(float *) fp1;
    float f2 = *(float *) fp2;
    if (f1 < f2) {
        return -1;
    } else {
        return (f1 > f2);
    }
}

void qsort_float(size_t n, float *base) {
    qsort(base, n, sizeof(float), cmp_float);
}

double median(const void *a, size_t n_elem, size_t size_elem,
        double (*accessor) (const void *)) {
    if (!n_elem) return 0.f;
    char *b = (char *) a;
    size_t N = n_elem + 1;
    if (N % 2) {
        return (accessor((const void *) b + size_elem * ((N / 2) - 1))
                + accessor((const void *) b + size_elem * (N / 2))) / 2;
    } else {
        return accessor((const void *) b + size_elem * ((N / 2) - 1));
    }
}

double first_quartile(const void *a, size_t n_elem, size_t size_elem,
        double (*accessor) (const void *)) {
    if (!n_elem) return 0.f;
    char *b = (char *) a;
    size_t N = n_elem + 3;
    switch (N % 4) {
        case 0:
            return accessor((const void *) b + size_elem * ((N / 4) - 1));
        case 1:
            return (3 * accessor((const void *) b + size_elem * ((N / 4) - 1))
                    + accessor((const void *) b + size_elem * (N / 4))) / 4;
        case 2:
            return (accessor((const void *) b + size_elem * ((N / 4) - 1))
                    + accessor((const void *) b + size_elem * (N / 4))) / 2;
        default:
            return (accessor((const void *) b + size_elem * ((N / 4) - 1))
                    + 3 * accessor((const void *) b + size_elem * (N / 4))) /4;
    }
}

double third_quartile(const void *a, size_t n_elem, size_t size_elem,
        double (*accessor) (const void *)) {
    if (!n_elem) return 0.f;
    char *b = (char *) a;
    size_t N = 3 * n_elem + 1;
    switch (N % 4) {
        case 0:
            return accessor((const void *) b + size_elem * ((N / 4) - 1));
        case 1:
            return (3 * accessor((const void *) b + size_elem * ((N / 4) - 1))
                    + accessor((const void *) b + size_elem * (N / 4))) / 4;
        case 2:
            return (accessor((const void *) b + size_elem * ((N / 4) - 1))
                    + accessor((const void *) b + size_elem * (N / 4))) / 2;
        default:
            return (accessor((const void *) b + size_elem * ((N / 4) - 1))
                    + 3 * accessor((const void *) b + size_elem * (N / 4))) /4;
    }
}

void solve_polynome(polynome_t *poly) {
    poly->solved = 1;
    long double a = poly->a;
    long double b = poly->b;
    long double c = poly->c;
    assert(a);
    long double delta = b*b - 4*a*c;
    poly->delta = delta;
    if (delta < 0.) return;
    poly->x1 = ( -b - pow(delta, 0.5) ) / (2 * a);
    poly->x2 = ( -b + pow(delta, 0.5) ) / (2 * a);
}

long binomial(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (n == 0 && k == 0) return 1;
    if (pascal.n_max == 0) {
        pthread_mutex_lock((pthread_mutex_t *) &pascal.m);
        if (pascal.n_max == 0) {
            pascal.t = calloc(n + 1, sizeof(long *));

            for (int i = 0; i < n + 1; ++i) {
                pascal.t[i] = calloc(i + 1, sizeof(long));
                pascal.t[i][0] = 1;
                for(int j = 1; j < i; ++j)
                    pascal.t[i][j] = pascal.t[i-1][j] + pascal.t[i-1][j-1];
                pascal.t[i][i] = 1;
            }
            pascal.n_max = n;
        }
        pthread_mutex_unlock((pthread_mutex_t *) &pascal.m);
    }

    if (pascal.n_max < n) {
        pthread_mutex_lock((pthread_mutex_t *) &pascal.m);
        if (pascal.n_max < n) {
            pascal.t = realloc(pascal.t, (n + 1)* sizeof(long *));
            for (int i = pascal.n_max; i < n + 1; ++i) {
                pascal.t[i] = calloc(i + 1, sizeof(long));
                pascal.t[i][0] = 1;
                for(int j = 1; j < i; ++j)
                    pascal.t[i][j] = pascal.t[i-1][j] + pascal.t[i-1][j-1];
                pascal.t[i][i] = 1;
            }
            pascal.n_max = n;
        }
        pthread_mutex_unlock((pthread_mutex_t *) &pascal.m);
    }

    return pascal.t[n][k] ;
}

int *copy_int_array(size_t n, int *a) {
    int *b = malloc(n * sizeof(int));
    memcpy(b, a, n * sizeof(int));
    return b;
}

int equals_int_array(size_t n, int *a, int *b) {
    int *end = a + n;
    while (a < end) if (*a++ != *b++) return 0;
    return 1;
}

#undef RAND
#undef L_RAND_MAX

