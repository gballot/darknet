#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#include "circular_buffer.h"
#include "protected_buffer.h"

protected_buffer_t *protected_buffer_init(int length) {
  protected_buffer_t *b;
  b = (protected_buffer_t *) malloc(sizeof(protected_buffer_t));
  b->buffer = circular_buffer_init(length);
  // Initialize the synchronization components
  pthread_mutex_init(&b->mutex, NULL);
  pthread_cond_init(&b->not_empty, NULL);
  pthread_cond_init(&b->not_full, NULL);
  return b;
}

void *protected_buffer_get(protected_buffer_t *b){
  void *d;
  // Enter mutual exclusion
  pthread_mutex_lock(&b->mutex);
  // Wait until there is a full slot to get data from the unprotected
  // circular buffer (circular_buffer_get).
  while(b->buffer->size <= 0)
      pthread_cond_wait(&b->not_empty, &b->mutex);
  // Signal or broadcast that an empty slot is available in the
  // unprotected circular buffer (if needed)
  if (b->buffer->size == b->buffer->max_size)
      pthread_cond_broadcast(&b->not_full);
  d = circular_buffer_get(b->buffer);
  // Leave mutual exclusion
  pthread_mutex_unlock(&b->mutex);
  
  return d;
}

void protected_buffer_put(protected_buffer_t *b, void *d){
  // Enter mutual exclusion
  pthread_mutex_lock(&b->mutex);
  // Wait until there is an empty slot to put data in the unprotected
  // circular buffer (circular_buffer_put).
  while(b->buffer->size >= b->buffer->max_size)
      pthread_cond_wait(&b->not_full, &b->mutex);
  // Signal or broadcast that a full slot is available in the
  // unprotected circular buffer (if needed)
  if (b->buffer->size == 0)
      pthread_cond_broadcast(&b->not_empty);
  circular_buffer_put(b->buffer, d);
  // Leave mutual exclusion
  pthread_mutex_unlock(&b->mutex);
}

void *protected_buffer_remove(protected_buffer_t *b){
  void *d;
  // Enter mutual exclusion
  pthread_mutex_lock(&b->mutex);
  // Signal or broadcast that an empty slot is available in the
  // unprotected circular buffer (if needed)
  if (b->buffer->size == b->buffer->max_size)
      pthread_cond_broadcast(&b->not_full);
  d = circular_buffer_get(b->buffer);
  // Leave mutual exclusion
  pthread_mutex_unlock(&b->mutex);
  return d;
}

int protected_buffer_add(protected_buffer_t *b, void *d){
  int done;
  // Enter mutual exclusion
  pthread_mutex_lock(&b->mutex);
  // Signal or broadcast that a full slot is available in the
  // unprotected circular buffer (if needed)
  if (b->buffer->size == 0)
      pthread_cond_broadcast(&b->not_empty);
  done = circular_buffer_put(b->buffer, d);
  // Leave mutual exclusion
  pthread_mutex_unlock(&b->mutex);
  return done;
}

void *protected_buffer_poll(protected_buffer_t *b, struct timespec *abstime){
  void *d = NULL;
  // Enter mutual exclusion
  pthread_mutex_lock(&b->mutex);
  // Wait until there is an empty slot to put data in the unprotected
  // circular buffer (circular_buffer_put) but waits no longer than
  // the given timeout.
  while (b->buffer->size <= 0) {
      if (pthread_cond_timedwait(&b->not_full, &b->mutex, abstime)
              == ETIMEDOUT) {
          pthread_mutex_unlock(&b->mutex);
          return NULL;
      }
  }
  // Signal or broadcast that a full slot is available in the
  // unprotected circular buffer (if needed)
  if (b->buffer->size == b->buffer->max_size)
      pthread_cond_broadcast(&b->not_full);
  d = circular_buffer_get(b->buffer);
  // Leave mutual exclusion
  pthread_mutex_unlock(&b->mutex);
  return d;
}

int protected_buffer_offer(protected_buffer_t *b, void *d,
        struct timespec *abstime){
  int done;
  // Enter mutual exclusion
  pthread_mutex_lock(&b->mutex);
  while(b->buffer->size >= b->buffer->max_size) {
      if (pthread_cond_timedwait(&b->not_full, &b->mutex, abstime)
              == ETIMEDOUT) {
          pthread_mutex_unlock(&b->mutex);
          return 0;
      }
  }
  // Signal or broadcast that a full slot is available in the
  // unprotected circular buffer (if needed) but waits no longer than
  // the given timeout.
  if (b->buffer->size == 0)
      pthread_cond_broadcast(&b->not_empty);
  done = circular_buffer_put(b->buffer, d);
  // Leave mutual exclusion
  pthread_mutex_unlock(&b->mutex);
  return done;
}
