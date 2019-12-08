#include <stdio.h>
#include <unistd.h>

#include "thread_pool.h"

thread_pool_t *thread_pool_init(int core_pool_size, int max_pool_size) {
  thread_pool_t *thread_pool;
  pthread_mutex_t mutex;
  pthread_mutex_init(&mutex, NULL);
  thread_pool = (thread_pool_t *) malloc(sizeof(thread_pool_t));
  thread_pool->core_pool_size = core_pool_size;
  thread_pool->max_pool_size = max_pool_size;
  thread_pool->size = 0;
  thread_pool->mutex = mutex;
  thread_pool ->thread_tab = malloc(sizeof(pthread_t) * max_pool_size);
  return thread_pool;
}

int pool_thread_create(thread_pool_t *thread_pool, run_func_t run,
        void *future, int force) {
  int done = 0;
  pthread_mutex_lock(&thread_pool->mutex);
  if (force) {
      if (thread_pool->size == thread_pool->max_pool_size) {
          pthread_mutex_unlock(&thread_pool->mutex);
          return -1;
      } else {
          pthread_t t;
          pthread_create(&t, NULL, run, future);
          thread_pool->thread_tab[thread_pool->size] = t;
          thread_pool->size++;
          pthread_mutex_unlock(&thread_pool->mutex);
          return done;
      }
  }
  // Always create a thread as long as there are less then
  // core_pool_size threads created.
  if (thread_pool->size < thread_pool->core_pool_size) {
      pthread_t t;
      pthread_create(&t, NULL, run, future);
      thread_pool->thread_tab[thread_pool->size] = t;
      thread_pool->size++;
      pthread_mutex_unlock(&thread_pool->mutex);
      return done;
  }
  pthread_mutex_unlock(&thread_pool->mutex);
  return -1;
}

void pool_thread_remove (thread_pool_t *thread_pool) {
  pthread_mutex_lock(&thread_pool->mutex);
  int null_found = 0;
  if (thread_pool->size > thread_pool->core_pool_size) {
      for (int i = 0 ; i < thread_pool->size ; i++) {
          if (null_found)
              thread_pool->thread_tab[i-1] = thread_pool->thread_tab[i];
          else if (thread_pool->thread_tab[i] == (pthread_t)NULL)
              null_found = 1;
      }
      if (null_found)
          thread_pool->size--;
  }
  pthread_mutex_unlock(&thread_pool->mutex);
}  

void wait_thread_pool_empty (thread_pool_t *thread_pool) {
  while(thread_pool->size > 0)
      sleep (20);
}  

