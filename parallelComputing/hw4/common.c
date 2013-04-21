#include "common.h"

#include <stdlib.h>
#include <sys/time.h>


double getTime() {
  struct timeval tv;
  struct timeval tz;
  double currTime;
  gettimeofday(&tv, &tz);
  currTime = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
  return currTime;
}


int binIndSearch(int *arr, int len , int val) {
  int ub, lb, mid;
  
  lb = 0;
  ub = len-1;

  while (lb <= ub) {
    mid = (ub+lb)/2;
    if (arr[mid] == val) {
      return mid;
    } else {
      if (val < arr[mid]) {
	ub = mid -1;
      } else {
	lb = mid + 1;
      }
    }
  }
  
  return -1;
}


void init(BVecComParams *bVecParams) {
  if (bVecParams != (BVecComParams*) 0) {
    bVecParams->numToSendProcs = -1;
    bVecParams->toSendProcs = (int *) 0;
    bVecParams->numToRecvProcs = -1;
    bVecParams->toRecvProcs = (int *) 0;
    bVecParams->sendPtr = (int *) 0;
    bVecParams->sendInd = (int *) 0;
    bVecParams->recvPtr = (int *) 0;
    bVecParams->recvInd = (int *) 0;
    bVecParams->sendBuf = (float *) 0;
    bVecParams->recvBuf = (float *) 0;
  }
}


void initCSRMat(CSRMat *csrMat) {
  if (NULL != csrMat) {
    csrMat->rowPtr = NULL;
    csrMat->colInd = NULL;
    csrMat->values = NULL;
  }  
}


void freeCSRMat(CSRMat *csrMat) {
  if (NULL != csrMat) {
    if (csrMat->rowPtr) {
      free(csrMat->rowPtr);
    }
    if (csrMat->colInd) {
      free(csrMat->colInd);
    }
    if (csrMat->values) {
      free(csrMat->values);
    }
    free(csrMat);
  }
}


void freeBVecComParams(BVecComParams *bVecParams) {
  if (bVecParams != (BVecComParams*) 0) {
    free(bVecParams->toSendProcs);
    free(bVecParams->toRecvProcs);
    free(bVecParams->sendPtr);
    free(bVecParams->sendInd);
    free(bVecParams->recvPtr);
    free(bVecParams->recvInd);
    free(bVecParams->sendBuf);
    free(bVecParams->recvBuf);
  }
}
