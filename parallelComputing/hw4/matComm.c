#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "matComm.h"
#include "io.h"
//sparse mat local to procs *myCSRMat

//*rowInfo store row details s.t. rowInfo[i] is starting row number &
//rowInfo[i+numProcs] is ending row number
void scatterMatrix(CSRMat *csrMat, CSRMat **myCSRMat, int **rowInfo) {
  int myRank, numProcs, i, prevEndRow;
  int totalRows, rowPerProc, extraRows;
  int startRow, endRow, myNNZCount;
  int status;

  //arrays to store distribution of rows and values for each proc
  int *rowCount, *colCount;

  //store displacement for scatterv relative to rowptr of csrMat
  int *dispRowPtr;
  
  //store sendCount for scatterv from rowptr of csrMat
  int *sendCountRowPtr;

  //store displacement for scatterv relative to colInd of csrMat
  int *dispColInd;

  //store sendCount for scatterv from colInd of csrMat
  int *sendCountColInd;

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  myNNZCount = -1;

  rowCount = (int *) 0;
  colCount = (int *) 0;
  
  dispRowPtr = (int *) 0;
  sendCountRowPtr = (int *) 0;
  
  dispColInd = (int *) 0;
  sendCountRowPtr = (int *) 0;
    
  printf("\nrank:%d divide indices", myRank);
  
  //divide row indices among matrix
  if (myRank == ROOT) {
    printf("\nrank:%d inside divide indices", myRank);
    //divide rows ind among matrix
    totalRows = csrMat->numRows;
    rowPerProc = totalRows / numProcs;
    extraRows = totalRows % numProcs;
    
    prevEndRow = -1;
    //assuming row ind starts from 0
    for (i = 0; i < numProcs; i++) {
      (*rowInfo)[i] = prevEndRow + 1;
      (*rowInfo)[i+numProcs] = (*rowInfo)[i] + rowPerProc - 1;
      if (extraRows-- > 0) {
	(*rowInfo)[i+numProcs] += 1;
      }
      prevEndRow = (*rowInfo)[i+numProcs];
    }

    for (i = 0; i < numProcs; i++) {
      printf("\nRank: %d First Row: %d Last Row: %d", i, (*rowInfo)[i],
	     (*rowInfo)[i+numProcs]);
    }
    
  }
  
  
  //send the row info to procs
  MPI_Bcast((*rowInfo), 2*numProcs, MPI_INT, ROOT, MPI_COMM_WORLD);
  
  //communicate matrix values to process
  if (myRank == ROOT) {
    //mark values and rows to be assigned to proc
    rowCount = (int *) malloc(sizeof(int)*numProcs);
    colCount = (int *) malloc(sizeof(int)*numProcs);
    
    dispRowPtr = (int *) malloc(sizeof(int)*numProcs);
    sendCountRowPtr = (int *) malloc(sizeof(int)*numProcs);
    
    dispColInd = (int *) malloc(sizeof(int)*numProcs);
    sendCountColInd = (int *) malloc(sizeof(int)*numProcs);
    
    for (i = 0; i < numProcs; i++) {
      //start end assigned to proc i
      startRow = (*rowInfo)[i];
      endRow = (*rowInfo)[i+numProcs];

      //row count for proc i
      rowCount[i] = endRow - startRow + 1;
      
      //TODO: verify below indices
      //col count or non-zero values for proc i
      colCount[i] = csrMat->rowPtr[endRow+1] - csrMat->rowPtr[startRow];
      
      //get offset and count of rows to be send to proc i
      dispRowPtr[i] = startRow;
      sendCountRowPtr[i] = rowCount[i];

      //get offset and count of values to be sent to proc i
      dispColInd[i] = csrMat->rowPtr[startRow];
      sendCountColInd[i] = colCount[i];

      //send nnz count to procs
      if (i != ROOT) {
	printf("\nrank:%d sending %d", myRank, *(colCount + i));
	MPI_Send(colCount+i, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
      } else {
	myNNZCount = colCount[ROOT];
      }
    }
    
    printf("\nRowCount: ");
    dispArray(rowCount, numProcs);

    printf("ColCount: ");
    dispArray(colCount, numProcs);

    printf("dispRowPtr: ");
    dispArray(dispRowPtr, numProcs);

    printf("sendCountRowPtr: ");
    dispArray(sendCountRowPtr, numProcs);

    printf("dispColInd: ");
    dispArray(dispColInd, numProcs);

    printf("sendCountColInd: ");
    dispArray(sendCountColInd, numProcs);

  } else {
    MPI_Recv(&myNNZCount, 1, MPI_INT, ROOT, 100, MPI_COMM_WORLD, &status);
  }

  printf("\nRank: %d nnzcount:%d ", myRank, myNNZCount);

  //prepare storage for local csr matrix
  /*myCSRMat = (CSRMat *) malloc(sizeof(CSRMat));
  (*myCSRMat)->nnzCount = myNNZCount;
  (*myCSRMat)->numRows = rowCount[myRank];
  (*myCSRMat)->numCols = csrMat->numCols;

  (*myCSRMat)->rowPtr = (int *) malloc(sizeof(int) * (*myCSRMat)->numRows);
  (*myCSRMat)->colInd = (int *) malloc(sizeof(int) * myNNZCount);
  (*myCSRMat)->values = (float *) malloc(sizeof(float) * myNNZCount);

  //communicate marked values to procs
  MPI_Scatterv(csrMat->rowPtr, sendCountRowPtr, dispRowPtr, MPI_INT,
	       (*myCSRMat)->rowPtr, (*myCSRMat)->numRows, MPI_INT, ROOT,
	       MPI_COMM_WORLD);
  MPI_Scatterv(csrMat->colInd, sendCountColInd, dispColInd, MPI_INT,
	       (*myCSRMat)->colInd, (*myCSRMat)->nnzCount, MPI_INT, ROOT,
	       MPI_COMM_WORLD);
  MPI_Scatterv(csrMat->rowPtr, sendCountColInd, dispColInd, MPI_FLOAT,
	       (*myCSRMat)->colInd, (*myCSRMat)->nnzCount, MPI_FLOAT, ROOT,
	       MPI_COMM_WORLD);
  */
  //free allocated mems
  
  if (rowCount && myRank == ROOT) {
    free(rowCount);
    rowCount = (int*)0;
 
    free(colCount);
    colCount = (int*)0;
 
    free(dispRowPtr);
    dispRowPtr = (int*)0;
 
    free(sendCountRowPtr);
    sendCountRowPtr = (int*)0;
 
    free(dispColInd);
    dispColInd = (int*)0;
 
    free(sendCountColInd);
    sendCountColInd = (int*)0;
  }
  
  printf("\n exiting matcomm rank: %d", myRank);
  
}
