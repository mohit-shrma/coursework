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
#include "mult.h"
#include "debug.h"



int main(int argc, char *argv[]) {

  char *matFileName, *vecFileName;
  int numProcs, myRank, i;
  CSRMat *csrMat, *myCSRMat;
  BVecComParams *myBVecParams;
  float *bVec;
  int dim, nnzCount, *rowInfo, myRowCount;
  float *myVec, *locProdVec;

  double totalTime, totalTimeStart, totalTimeEnd;
  double vecCommTime, vecCommTimeStart, vecCommTimeEnd;

  //TODO: remove this after verification as can replace myVec with 
  //product itself
  float *prodVec;

  FILE *myLogFile, *mpiResFile, *serResFile;
  
  char strTemp[100], opFileName[20];

  MPI_Init(&argc, &argv);
  
  if (argc <= 2) {
    printf("%s: insufficient arguments\n", argv[0]);
    exit(1);
  }
  
  matFileName = argv[1];
  vecFileName = argv[2];

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  //set pointers to NULL
  csrMat = NULL;//(CSRMat *) 0;
  rowInfo = NULL;//(int*) 0;
  myCSRMat = NULL;//(CSRMat *) 0; 
  bVec = NULL;//(float *) 0;
  myVec = NULL;//(float *) 0;
  myBVecParams = NULL;//(BVecComParams *) 0;
  locProdVec = NULL;
  prodVec = NULL;
  myLogFile = NULL;
  mpiResFile = NULL;
  serResFile = NULL;

  myCSRMat = (CSRMat *) malloc(sizeof(CSRMat));
  initCSRMat(myCSRMat);

  myBVecParams = (BVecComParams *) malloc(sizeof(BVecComParams));
  rowInfo = (int*) malloc(sizeof(int)*2*numProcs);
  memset(rowInfo, 0, sizeof(int)*2*numProcs);
  
  //initialize the log file
  if (DEBUG) {
    sprintf(strTemp, "%d", myRank);
    strcat(strTemp, "_spMatVec.log");
    myLogFile = fopen(strTemp, "w");
    dbgPrintf(myLogFile, "\n rank:%d numProcs:%d ", myRank, numProcs);
  }


  //read sparse matrix
  if (myRank == ROOT) {

    //get dimension
    getDimNCount(matFileName, &dim, &nnzCount);
    
    //read the matrix
    csrMat = readSparseMat(matFileName, dim, nnzCount);
    if (DEBUG) {
      dbgPrintf(myLogFile, "\n display full sparse mat rank:%d numProcs:%d\n",
		myRank, numProcs);
      logSparseMat(csrMat, myRank, myLogFile);
    }
  } else {
    csrMat = (CSRMat *) malloc(sizeof(CSRMat));
    initCSRMat(csrMat);
  }
  
  dbgPrintf(myLogFile, "\nrank:%d before matrix scatter\n", myRank);
  scatterMatrix(csrMat, &myCSRMat, rowInfo);
  
  if (DEBUG) {
    dbgPrintf(myLogFile, "\nlocal sparse mat rank:%d\n", myRank);
    logSparseMat(myCSRMat, myRank, myLogFile);
    
    dbgPrintf(myLogFile, "\n rowInfo: ");
    logArray(rowInfo, numProcs*2, myRank, myLogFile);
  }

  //read vector
  if (myRank == ROOT) {
    //allocate space for vector to read
    bVec = (float* ) malloc(sizeof(float) * dim);
    
    //also alllocate space for product vector, this is not required as can used original 
    //input vector to store final product vector
    prodVec = (float *) malloc(sizeof(float) * dim);
    memset(bVec, 0, sizeof(float)*dim);

    //read sparse vector
    readSparseVec(bVec, vecFileName, dim);

    if (DEBUG) {
      dbgPrintf(myLogFile, "\n display sparse vector:");
      logFArray(bVec, dim, myRank, myLogFile);
    }
  }

 
  //allocate space for local part of vector
  myRowCount = rowInfo[myRank+numProcs] - rowInfo[myRank] + 1;
  dbgPrintf(myLogFile, "\nrank: %d rowCount:%d", myRank, myRowCount);
  myVec = (float *) malloc(sizeof(float) * myRowCount);


  //scatter vector to processors
  dbgPrintf(myLogFile, "\nrank:%d before vector scatter\n", myRank);
  scatterVector(bVec, rowInfo, myVec);

  if (DEBUG) {
    dbgPrintf(myLogFile, "\nlocal vec rank:%d\n", myRank);
    logFArray(myVec, myRowCount, myRank, myLogFile);
    dbgPrintf(myLogFile, "\nrank:%d before vector comm\n", myRank);
  }

  //allocate space for local product vector
  locProdVec = (float*) malloc(sizeof(float) * myRowCount);
  memset(locProdVec, 0, sizeof(float) * myRowCount);

  //start the total time timer
  totalTimeStart = getTime();

  //identify & communicate only required values of vector
  prepareVectorComm(myCSRMat, myVec, myBVecParams, rowInfo, &vecCommTimeStart);
  
  dbgPrintf(myLogFile, "\nrank:%d after vector comm\n", myRank);
  
  //perform multiplication with required values of vector
  computeLocalProd(myCSRMat, myBVecParams, myVec, locProdVec, myRank);

  dbgPrintf(myLogFile, "\nrank:%d after local prod gen\n", myRank);

  vecCommTimeEnd = getTime();
  totalTimeEnd = getTime();

  //compute total time taken
  totalTime = totalTimeEnd - totalTimeStart;
  vecCommTime = vecCommTimeEnd - vecCommTimeStart;
  
  if (myRank == ROOT) {
    printf("\nTotal time taken : %f sec", totalTime);
    printf("\nTime taken (steps 5 & 6) : %f sec\n", vecCommTime);
  }
  

  //gather the results of multiplication at root, overwrite myVec with results
  gatherVector(locProdVec, rowInfo, prodVec);

  dbgPrintf(myLogFile, "\n after gathering computed subProduct across nodes");
 
  if (myRank == ROOT) {
    //write the resulting mpi parallel product vector to a file
    //initialize the result file for the mpi version
    opFileName[0] = '\0';
    sprintf(strTemp, "%d", dim);
    strcat(opFileName, "o");
    strcat(opFileName, strTemp);
    strcat(opFileName, ".vec");
    mpiResFile = fopen(opFileName, "w");
    dbgPrintf(myLogFile, "\nwriting to mpi prod file");
    //write down the results of mpi parallel multiplication
    for (i = 0; i < dim; i++) {
      fprintf(mpiResFile, "%f\n", prodVec[i]);
    }
    dbgPrintf(myLogFile, "\nwritten to mpi prod file");
    fclose(mpiResFile);
  }
  
  /*
  if (myRank == ROOT) {
    //reset prod vec which wil contain serial results
    memset(prodVec, 0, (sizeof(float))*dim);

    //perform serial multiplication at root
    computeSerialProd(csrMat, bVec, prodVec);

    //initialize the result file for serial version
    sprintf(strTemp, "%d", myRank);
    strcat(strTemp, "_non_mpi_res.log");
    serResFile = fopen(strTemp, "w");
    dbgPrintf(myLogFile, "\nwriting to serial file");
    //write down the results of serial multiplication
    for (i = 0; i < dim; i++) {
      fprintf(serResFile, "%f\n", prodVec[i]);
    }
    dbgPrintf(myLogFile, "\nwritten to serial file");
    fclose(serResFile);
  }*/
  


  if (NULL != rowInfo) {
    free(rowInfo);
    dbgPrintf(myLogFile, "\n free rowInfo");
  }

  if (NULL != myCSRMat) {
    freeCSRMat(myCSRMat);
    dbgPrintf(myLogFile, "\n free myCSRMat");
  }

  if (NULL != csrMat) {
    freeCSRMat(csrMat);
    dbgPrintf(myLogFile, "\n free csrmat");
  }

  if (NULL != myBVecParams) {
    freeBVecComParams(myBVecParams);
    dbgPrintf(myLogFile, "\n free myBVecParams");
  }
  
  if (NULL != bVec) {
    free(bVec);
    dbgPrintf(myLogFile, "\n free bvec");
  }

  if (NULL != myVec) {
    free(myVec);
    dbgPrintf(myLogFile, "\n free myVec\n");
  }

  if (NULL != locProdVec) {
    free(locProdVec);
    dbgPrintf(myLogFile, "\n free locProdVec\n");
  }

  if (NULL != prodVec) {
    free(prodVec);
    dbgPrintf(myLogFile, "\n free prodVec\n");
  }

  if (myLogFile) {
    fclose(myLogFile);
  }

  MPI_Finalize();

  return 0;
}

