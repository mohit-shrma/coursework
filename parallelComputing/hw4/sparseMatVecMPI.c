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

  csrMat = (CSRMat *) 0;
  rowInfo = (int*) 0;
  myCSRMat = (CSRMat *) 0; 
  bVec = (float *) 0;
  myVec = (float *) 0;
  myBVecParams = (BVecComParams *) 0;

  myCSRMat = (CSRMat *) malloc(sizeof(CSRMat));
  myBVecParams = (BVecComParams *) malloc(sizeof(BVecComParams));
  rowInfo = (int*) malloc(sizeof(int)*2*numProcs);
  memset(rowInfo, 0, sizeof(int)*2*numProcs);
    
  printf("\n rank:%d numProcs:%d ", myRank, numProcs);
  
  //read sparse matrix and vector
  if (myRank == ROOT) {
    //get dimension
    getDimNCount(matFileName, &dim, &nnzCount);
    
    //read the matrix
    csrMat = readSparseMat(matFileName, dim, nnzCount);
    printf("\n display full sparse mat rank:%d numProcs:%d\n", myRank,
	   numProcs);
    displSparseMat(csrMat, myRank);
    
    //read the vector
    bVec = (float* ) malloc(sizeof(float) * dim);
    
    printf("\n display full sparse mat rank:%d numProcs:%d\n", myRank,
	   numProcs);
    displSparseMat(csrMat, myRank);
    
    //readSparseVec(bVec, vecFileName, dim);
    //printf("\n display sparse vector:");
    //dispFArray(bVec, dim, myRank);
  } else {
    csrMat = (CSRMat *) malloc(sizeof(CSRMat));
  }

  
  printf("\nrank:%d before matrix scatter\n", myRank);
  scatterMatrix(csrMat, &myCSRMat, &rowInfo);


  printf("\nlocal sparse mat rank:%d\n", myRank);
  displSparseMat(myCSRMat, myRank);
  
  /*
  myRowCount = rowInfo[myRank+numProcs] - rowInfo[myRank] + 1;
  myVec = (float *) malloc(sizeof(float) * myRowCount);
  
  printf("\nrank:%d before vector scatter\n", myRank);
  scatterVector(bVec, rowInfo, myVec);
  printf("\nlocal vec rank:%d\n", myRank);
  dispFArray(myVec, myRowCount, myRank);
  */
  //prepareVectorComm(myCSRMat, myVec, myBVecParams, rowInfo);
  

  /*
  if (rowInfo) {
    free(rowInfo);
  }

  if (myCSRMat) {
    free(myCSRMat);
  }

  if (csrMat) {
    free(csrMat);
  }
  
  if (bVec) {
    free(bVec);
  }

  */
  
  MPI_Finalize();

  return 0;
}
