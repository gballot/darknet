#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <pthread.h>

#include "thread_pool.h"
#include "protected_buffer.h"

struct _executor_t;

typedef struct {
  void               *params;
  run_func_t          run;
  long                period;
  struct _executor_t *executor;
} callable_t;

typedef struct {
  int             completed;
  callable_t     *callable;
  void           *result;
  pthread_mutex_t mutex;
  pthread_cond_t  var;
} future_t;

typedef struct _executor_t {
  thread_pool_t      *thread_pool;
  long                keep_alive_time;
  protected_buffer_t *futures;
} executor_t;

extern executor_t *executor_init(int  core_pool_size, int  max_pool_size,
        long keep_alive_time, int  callable_array_size);

extern future_t *submit_callable(executor_t * executor, callable_t * callable);

// Get result from callalbe execution. Block if not available.
void * get_callable_result(future_t * future);

// Wait for pool threads to be completed
void executor_shutdown(executor_t * executor);

#endif
