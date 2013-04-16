/*
figure out how to communicate only the required vector elements
*/

#ifndef _VEC_COMM_
#define _VEC_COMM_
#include "common.h"

void addToSet(int *set, int val);
int isInSet(int *set, int val);
int sizeSet(int *set, int capacity);
int* getSetElements(int *set, int capacity);
void prepareVectorComm(CSRMat* myCSRMat, BVecComParams *bVecParams);

#endif
