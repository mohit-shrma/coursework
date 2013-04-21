#ifndef _DEBG_
#define _DEBG_

#include <stdio.h>
#include <stdarg.h>

#define DEBUG 0

void dbgPrintf(FILE* file, const char *fmt, ...);

#endif
