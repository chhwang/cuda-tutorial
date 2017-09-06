/* Author: Changho Hwang         */
/* E-mail: ch.hwang128@gmail.com */

/* common/cpu_timer.h */

#ifndef CPUTIMER_H
#define CPUTIMER_H

#ifdef __cplusplus
extern "C" {
#endif

long cputimer_nsec(void);
double cputimer_sec(void);

#ifdef __cplusplus
}
#endif

#endif /* CPUTIMER_H */
