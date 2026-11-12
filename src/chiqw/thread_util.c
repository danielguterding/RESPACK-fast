/* This file is part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
 * GNU General Public License, version 3 (see GPL/GPLv3.txt).
 *
 * One background thread for m_eps_async (post-processing of a q point while the next one is computed). */
#include <pthread.h>
#include <stddef.h>

static pthread_t th;
static void (*fn)(void);

static void *run(void *arg) {
  (void)arg;
  fn();
  return NULL;
}

/* start f() in a new thread; 0 on success */
int respack_thread_start(void (*f)(void)) {
  pthread_attr_t a;
  int rc;
  fn = f;
  pthread_attr_init(&a);
  pthread_attr_setstacksize(&a, (size_t)256 << 20); /* Fortran automatic arrays */
  rc = pthread_create(&th, &a, run, NULL);
  pthread_attr_destroy(&a);
  return rc;
}

int respack_thread_join(void) { return pthread_join(th, NULL); }
