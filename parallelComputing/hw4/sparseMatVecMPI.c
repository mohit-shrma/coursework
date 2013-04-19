/*
 * implements sparse matrix vector multiplication A*b = x
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


#include "common.h"
#include "io.h"
#include "matComm.h"
#include "vecComm.h"


int main(int argc, char *argv[]) {
  
  char *matFileName, *vecFileName;
  int numProcs, myRank;
  CSRMat *csrMat, *myCSRMat;
  BVecComParams *myBVecParams;
  float *bVec;
  int dim, nnzCount, *rowInfo, myRowCount;
  float *myVec;

  MPI_Init(&argc, &argv);
  
  if (argc <= 2) {
    printf("%s: insufficient arguments\n", argv[0]);
    exit(1);
  }
  

  matFileName = argv[1];
  vecFileName = argv[2];

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  csrMat = NULL;//(CSRMat *) 0;
  rowInfo = NULL;//(int*) 0;
  myCSRMat = NULL;//(CSRMat *) 0; 
  bVec = NULL;//(float *) 0;
  myVec = NULL;//(float *) 0;
  myBVecParams = NULL;//(BVecComParams *) 0;

  myCSRMat = (CSRMat *) malloc(sizeof(CSRMat));
  initCSRMat(myCSRMat);

  myBVecParams = (BVecComParams *) malloc(sizeof(BVecComParams));
  rowInfo = (int*) malloc(sizeof(int)*2*numProcs);
  memset(rowInfo, 0, sizeof(int)*2*numProcs);
    
  printf("\n rank:%d numProcs:%d ", myRank, numProcs);
  
  //read sparse matrix
  if (myRank == ROOT) {
    //get dimension
    getDimNCount(matFileName, &dim, &nnzCount);
    
    //read the matrix
    csrMat = readSparseMat(matFileName, dim, nnzCount);
    printf("\n display full sparse mat rank:%d numProcs:%d\n", myRank,
	   numProcs);
    displSparseMat(csrMat, myRank);
  } else {
    csrMat = (CSRMat *) malloc(sizeof(CSRMat));
    initCSRMat(csrMat);
  }
  
  printf("\nrank:%d before matrix scatter\n", myRank);
  scatterMatrix(csrMat, &myCSRMat, rowInfo);

  printf("\nlocal sparse mat rank:%d\n", myRank);
  displSparseMat(myCSRMat, myRank);

  
  //read vector
  if (myRank == ROOT) {
    //read the vector
    bVec = (float* ) malloc(sizeof(float) * dim);
    memset(bVec, 0, sizeof(float)*dim);
    readSparseVec(bVec, vecFileName, dim);
    printf("\n display sparse vector:");
    dispFArray(bVec, dim, myRank);
  }

 
  //allocate space for local part of vector
  myRowCount = rowInfo[myRank+numProcs] - rowInfo[myRank] + 1;
  myVec = (float *) malloc(sizeof(float) * myRowCount);
 
  
  //scatter vector
  printf("\nrank:%d before vector scatter\n", myRank);
  scatterVector(bVec, rowInfo, myVec);

  printf("\nlocal vec rank:%d\n", myRank);
  dispFArray(myVec, myRowCount, myRank);
  
  prepareVectorComm(myCSRMat, myVec, myBVecParams, rowInfo);
    

  
  if (NULL != rowInfo) {
    free(rowInfo);
  }

  if (NULL != myCSRMat) {
    freeCSRMat(myCSRMat);
  }

  if (NULL != csrMat) {
    freeCSRMat(csrMat);
  }
  
  if (NULL != bVec) {
    free(bVec);
  }

  if (NULL != myVec) {
    free(myVec);
  }
  
  MPI_Finalize();

  return 0;
}
