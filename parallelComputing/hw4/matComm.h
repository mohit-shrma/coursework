#ifndef _MATT_COMM_
#define _MATT_COMM_
#include "common.h"

void scatterMatrix(CSRMat *csrMat, CSRMat **myCSRMat, int **rowInfo);

void scatterVector(int *vec, int *rowInfo);

#endif
