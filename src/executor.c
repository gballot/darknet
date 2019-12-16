#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "executor.h"
#include "utils.h"

static void *callable_run (void *arg);

executor_t *executor_init (int core_pool_size, int max_pool_size, 
        long keep_alive_time, int callable_array_size) {
    executor_t *executor;
    executor = (executor_t *) malloc(sizeof(executor_t));
    executor->keep_alive_time = keep_alive_time;
    executor->thread_pool = thread_pool_init(core_pool_size, max_pool_size);
    executor->futures = protected_buffer_init(callable_array_size);
    return executor;
}

future_t *submit_callable (executor_t *executor, callable_t *callable) {
    future_t *future = (future_t *) malloc(sizeof(future_t));
    callable->executor = executor;
    future->callable  = callable;
    future->completed = 0;
    pthread_mutex_init(&future->mutex, NULL);
    pthread_cond_init(&future->var, NULL);
    if(executor->thread_pool->size < executor->thread_pool->core_pool_size) {
        pool_thread_create(executor->thread_pool, callable_run, future, 0);
    } else {
        int try_put = protected_buffer_add(executor->futures, future);
        if (try_put == 0) {
            int try_force = pool_thread_create(executor->thread_pool,
                    callable_run, future, 1);
            if (try_force == -1) {
                free(future);
                return NULL;
            }
        }
    }
    return future;
}

future_t *submit_callable_blocking(executor_t *executor, callable_t *callable){
    future_t *future = (future_t *) malloc(sizeof(future_t));
    callable->executor = executor;
    future->callable  = callable;
    future->completed = 0;
    pthread_mutex_init(&future->mutex, NULL);
    pthread_cond_init(&future->var, NULL);
    if(executor->thread_pool->size < executor->thread_pool->core_pool_size) {
        pool_thread_create(executor->thread_pool, callable_run, future, 0);
    } else {
        fprintf(stderr, "try put...\n");
        protected_buffer_put(executor->futures, future);
        fprintf(stderr, "put ok.\n");
    }
    return future;
}

void *get_callable_result (future_t *future) {
    void *result;
    // Protect from concurrent access
    pthread_mutex_lock(&future->mutex);
    // Wait for the result
    while(future->completed == 0)
        pthread_cond_wait(&future->var, &future->mutex);
    result = (void *) future->result;
    // deallocate future
    pthread_mutex_unlock(&future->mutex);
    free(future);
    return result;
}

static void *callable_run (void *arg) {
    future_t *future = (future_t *) arg;
    executor_t *executor = (executor_t *) future->callable->executor;
    struct timespec ts_deadline;
    struct timeval tv_deadline; 
    pthread_mutex_t periodic_mutex;
    pthread_cond_t periodic_cond;

    while (1) {
        while (1) {
            // Protect from concurrent access
            pthread_mutex_lock(&future->mutex);
            // When the callable is not periodic, leave first inner loop
            if (future->callable->period == 0) {
                future->result =
                    future->callable->run(future->callable->params);
                future->completed = 1;
                pthread_cond_broadcast(&future->var);
                pthread_mutex_unlock(&future->mutex);
                break;
            } else {
                gettimeofday (&tv_deadline, NULL);
                TIMEVAL_TO_TIMESPEC(&tv_deadline, &ts_deadline);
                add_millis_to_timespec(&ts_deadline, future->callable->period);
                future->result =
                    future->callable->run(future->callable->params);
                // waits for ts_deadline
                pthread_mutex_lock(&periodic_mutex);
                pthread_cond_timedwait(&periodic_cond, &periodic_mutex,
                        &ts_deadline);
                pthread_mutex_unlock(&periodic_mutex);
                pthread_mutex_unlock(&future->mutex);
            }
        }
        if (executor->keep_alive_time != 0) {
            future = NULL;
            // If the executor is configured to release threads when they
            // are idle for keep_alive_time milliseconds, try to get a new
            // callable / future for at most keep_alive_time milliseconds.
            struct timespec timeToWait;
            struct timeval now;
            gettimeofday(&now,NULL);
            TIMEVAL_TO_TIMESPEC (&now, &timeToWait);
            add_millis_to_timespec(&timeToWait, executor->keep_alive_time);
            future = (future_t *) protected_buffer_poll(executor->futures,
                    &timeToWait);
            if (future == NULL) {
                pool_thread_remove(executor->thread_pool);
                break;
            }
        } else {
            // If the executor does not realease inactive thread, just wait
            // and block for the next available callable / future.
            future = (future_t *) protected_buffer_get(executor->futures);
        }
    }
    return NULL;
}

// Wait for pool threads to be completed
void executor_shutdown (executor_t *executor) {
    wait_thread_pool_empty(executor->thread_pool);
}

