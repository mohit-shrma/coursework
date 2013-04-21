#ifndef _MULT_
#define _MULT_
#include "common.h"

void computeLocalProd(CSRMat *myCSRMat, BVecComParams *myVecParams,
		      float *localVec, float *locProdvec, int myRank);

void computeSerialProd(CSRMat *csrMat, float *fullVec, float *prodVec);

#endif
