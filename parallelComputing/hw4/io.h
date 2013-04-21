#ifndef _IO_
#define _IO_

#include "common.h"

#define BUF_SZ 200

CSRMat* readSparseMat(char *matFileName, int dim, int nnz); 

void readSparseVec(float *bVec, char* vecFileName, int dim);

void getDimNCount(char *vecFileName, int *dim, int *nnz);

void displSparseMat(CSRMat *csrMat, int rank);

void logSparseMat(CSRMat *csrMat, int rank, FILE* myLogFile);

void dispArray(int *arr, int len, int rank);

void dispFArray(float *arr, int len, int rank);

void logArray(const int *arr, int len, const int rank, FILE *logFile);

void logFArray(const float *arr, int len, const int rank, FILE *logFile);

#endif


