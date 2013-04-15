/*
 * implements sparse matrix vector multiplication A*b = x
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
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

  numProcs = MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  myRank = MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  myCSRMat = (CSRMat *) malloc(sizeof(CSRMat));
  rowInfo = (int*) malloc(sizeof(int)*numProcs);



  //read and print sparse matrix
  if (myRank == ROOT) {
    getDimNCount(matFileName, &dim, &nnzCount);
    csrMat = readSparseMat(matFileName, dim, nnzCount);
    printf("\n display full sparse mat rank:%d numProcs:%d\n", myRank,
	   numProcs);
    displSparseMat(csrMat);
  } else {
  }
  
  scatterMatrix(csrMat, &myCSRMat, &rowInfo);
  printf("\n sparse mat rank:%d\n", myRank);
  displSparseMat(myCSRMat);
  

  return 0;
}
