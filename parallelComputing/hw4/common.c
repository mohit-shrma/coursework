#include "common.h"
#include <stdlib.h>

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
