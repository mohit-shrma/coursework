#include "debug.h"

void dbgPrintf(FILE *file, const char *fmt, ...) {
  if (DEBUG) {
    va_list argPtr;
    va_start(argPtr, fmt);
    vfprintf(file, fmt, argPtr);
    va_end(argPtr);
  }
}
