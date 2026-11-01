#define _GNU_SOURCE
#include <pthread.h>
int pthread_setname_np(pthread_t t, const char *n) {
  (void)t; (void)n; return 0;
}
