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



//distribute the matrices and vector from file to each process
void distribute(char *matFileName, char *vecFileName, int n) {
  
  //get the number of procs

  //get the current rank
  
  //start row ind to store

  //end row ind to store

  //initialize the local CSRMat struct variable

  //read the matrix file: row, col, val
  
  //fill the local CSRMat struct var 

  //read the vector file

  //fill the local vector
  
}


int main(int argc, char *argv[]) {
  
  char *matFileName, *vecFileName;
  int numProcs, myRank;
  CSRMat *csrMat, *myCSRMat;
  int dim, nnzCount, *rowInfo;

  MPI_Init(&argc, &argv);
  
  if (argc <= 2) {
    printf("%s: insufficient arguments\n", argv[0]);
    exit(1);
  }
  

  matFileName = argv[1];
  vecFileName = argv[2];

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  csrMat = (CSRMat *)0;
  rowInfo = (int*) 0;
  myCSRMat = (CSRMat *) 0; 
    
  myCSRMat = (CSRMat *) malloc(sizeof(CSRMat));
  rowInfo = (int*) malloc(sizeof(int)*2*numProcs);
  memset(rowInfo, 0, sizeof(int)*2*numProcs);
    
  printf("\n rank:%d numProcs:%d ", myRank, numProcs);

  
  //read and print sparse matrix
  if (myRank == ROOT) {
    getDimNCount(matFileName, &dim, &nnzCount);
    csrMat = readSparseMat(matFileName, dim, nnzCount);
    //printf("\n display full sparse mat rank:%d numProcs:%d\n", myRank,
    //numProcs);
    //displSparseMat(csrMat);
  } else {
    csrMat = (CSRMat *) malloc(sizeof(CSRMat));
  }


  printf("\nrank:%d before scatter\n", myRank);
  scatterMatrix(csrMat, &myCSRMat, &rowInfo);
  printf("\nlocal sparse mat rank:%d\n", myRank);
  displSparseMat(myCSRMat, myRank);

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
  */
  
  MPI_Finalize();

  return 0;
}
