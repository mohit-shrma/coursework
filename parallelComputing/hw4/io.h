#ifndef _IO_
#define _IO_

#include "common.h"

#define BUF_SZ 200

CSRMat* readSparseMat(char *matFileName, int dim, int nnz); 

float* readSparseVec(char* vecFileName, int dim);

void getDimNCount(char *vecFileName, int *dim, int *nnz);

void displSparseMat(CSRMat *csrMat, int rank);

void dispArray(int *arr, int len, int rank);

void dispFArray(float *arr, int len, int rank);

#endif


