#ifndef _MATT_COMM_
#define _MATT_COMM_
#include "common.h"

void scatterMatrix(CSRMat *csrMat, CSRMat **myCSRMat, int *rowInfo);

void scatterVector(float *vec, int *rowInfo, float *myVec);

void gatherVector(float *localProdVec, int *rowInfo, float *prodVec);

#endif
