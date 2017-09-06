/* Author: Changho Hwang         */
/* E-mail: ch.hwang128@gmail.com */

/* common/cputimer.c */

#include <time.h>
#include <stdio.h>

long cputimer_nsec(void)
{
    struct timespec tspec;
    if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
        fprintf(stderr, "cputimer error: clock_gettime failed.\n");
        return -1;
    }
	return (tspec.tv_sec * (1000000000L) + tspec.tv_nsec);
}

double cputimer_sec(void)
{
    struct timespec tspec;
    if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
        fprintf(stderr, "cputimer error: clock_gettime failed.\n");
        return -1;
    }
	return (tspec.tv_nsec / 1000000000.0 + tspec.tv_sec);
}
