#ifndef _IO_
#define _IO_

#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define BUF_SZ 200

CSRMat* readSparseMat(char *matFileName, int dim, int nnz); 

int* readSparseVec(char* vecFileName, int dim);

void getDimNCount(char *vecFileName, int *dim, int *nnz);

void displSparseMat(CSRMat *csrMat);

void dispArray(int *arr, int len);

#endif


