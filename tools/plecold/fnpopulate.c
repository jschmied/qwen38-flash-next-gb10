// FNPOPULATE helper (jschmied 2026-09-25, local): populate a list of 4 KiB pages from N pthreads, GIL-free.
// int fn_populate(const unsigned long *pages, int n, int threads): MADV_POPULATE_READ per page, returns #errors.
#include <pthread.h>
#include <sys/mman.h>
#include <stdatomic.h>
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
typedef struct { const unsigned long *pages; int n; atomic_int next; atomic_int errors; } job_t;
static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    for (;;) {
        int i = atomic_fetch_add(&j->next, 1);
        if (i >= j->n) break;
        if (madvise((void *)j->pages[i], 4096, MADV_POPULATE_READ) != 0) atomic_fetch_add(&j->errors, 1);
    }
    return 0;
}
int fn_populate(const unsigned long *pages, int n, int threads) {
    if (n <= 0) return 0;
    if (threads > n) threads = n;
    if (threads > 64) threads = 64;
    job_t j = {pages, n, 0, 0};
    pthread_t t[64];
    int started = 0;
    for (int k = 1; k < threads; k++) if (pthread_create(&t[started], 0, worker, &j) == 0) started++;
    worker(&j);
    for (int k = 0; k < started; k++) pthread_join(t[k], 0);
    return atomic_load(&j.errors);
}
