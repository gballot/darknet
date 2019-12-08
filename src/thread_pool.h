#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <pthread.h>

#include "protected_buffer.h"

typedef void *(*run_func_t)(void *);

typedef struct {
  int             core_pool_size;
  int             max_pool_size;
  int             size;
  pthread_mutex_t mutex;
  pthread_t      *thread_tab;
} thread_pool_t;

/**
 * Create a thread pool. This pool must be protected against
 * concurrent accesses.
 */
extern thread_pool_t *thread_pool_init(int core_pool_size, int max_pool_size);

/**
 * Create a thread. If the threads number created is not greater than
 * core_pool_size, create a new thread. If it is and force is true,
 * create a new thread. If a thread is created, increment the thread
 * number. Use run as a main procedure and future as run parameter.
 */
extern int pool_thread_create(thread_pool_t *thread_pool, run_func_t run,
        void *executor, int force);

/**
 * Decrease thread number and broadcast update.
 */
extern void pool_thread_remove(thread_pool_t *thread_pool);

/**
 * Wait until thread number equals zero.
 */
extern void wait_thread_pool_empty(thread_pool_t *thread_pool);
#endif
