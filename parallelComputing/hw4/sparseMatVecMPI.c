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

  FILE *myLogFile;
  char strTemp[100];

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
  
  //initialize the log file
  sprintf(strTemp, "%d", myRank);
  strcat(strTemp, "_spMatVec.log");
  myLogFile = fopen(strTemp, "w");

  fprintf(myLogFile, "\n rank:%d numProcs:%d ", myRank, numProcs);
  
  //read sparse matrix
  if (myRank == ROOT) {
    //get dimension
    getDimNCount(matFileName, &dim, &nnzCount);
    
    //read the matrix
    csrMat = readSparseMat(matFileName, dim, nnzCount);
    fprintf(myLogFile, "\n display full sparse mat rank:%d numProcs:%d\n", myRank,
	   numProcs);
    logSparseMat(csrMat, myRank, myLogFile);
  } else {
    csrMat = (CSRMat *) malloc(sizeof(CSRMat));
    initCSRMat(csrMat);
  }
  /*
  fprintf(myLogFile, "\nrank:%d before matrix scatter\n", myRank);
  scatterMatrix(csrMat, &myCSRMat, rowInfo);
  
  fprintf(myLogFile, "\nlocal sparse mat rank:%d\n", myRank);
  logSparseMat(myCSRMat, myRank, myLogFile);
  */
  
  //read vector
  if (myRank == ROOT) {
    //read the vector
    bVec = (float* ) malloc(sizeof(float) * dim);
    memset(bVec, 0, sizeof(float)*dim);
    readSparseVec(bVec, vecFileName, dim);
    fprintf(myLogFile, "\n display sparse vector:");
    logFArray(bVec, dim, myRank, myLogFile);
  }

 
  //allocate space for local part of vector
  myRowCount = rowInfo[myRank+numProcs] - rowInfo[myRank] + 1;
  fprintf(myLogFile, "\nrank: %d rowCount:%d", myRank, myRowCount);
  myVec = (float *) malloc(sizeof(float) * myRowCount);
 
  
  //scatter vector
  fprintf(myLogFile, "\nrank:%d before vector scatter\n", myRank);
  scatterVector(bVec, rowInfo, myVec);

  fprintf(myLogFile, "\nlocal vec rank:%d\n", myRank);
  logFArray(myVec, myRowCount, myRank, myLogFile);
  
  //communicate only required values of vector
  //prepareVectorComm(myCSRMat, myVec, myBVecParams, rowInfo);
  
  //perform multiplication with required values of vector

  //gather the results of multiplication at root

    
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

  fclose(myLogFile);
  
  MPI_Finalize();

  return 0;
}
