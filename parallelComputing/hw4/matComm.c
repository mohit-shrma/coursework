#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "matComm.h"
#include "io.h"
//sparse mat local to procs *myCSRMat

//*rowInfo store row details s.t. rowInfo[i] is starting row number &
//rowInfo[i+numProcs] is ending row number
void scatterMatrix(CSRMat *csrMat, CSRMat **myCSRMat, int *rowInfo) {
  int myRank, numProcs, i, prevEndRow;
  int totalRows, rowPerProc, extraRows;
  int startRow, endRow, myNNZCount;
  int status;
  FILE *myLogFile;
  char strTemp[100];
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

  rowCount = NULL;
  colCount = NULL;
  
  dispRowPtr = NULL;
  sendCountRowPtr = NULL;
  
  dispColInd = NULL;
  sendCountRowPtr = NULL;
  sendCountColInd = NULL;


  MPI_Bcast(rowInfo, 2*numProcs, MPI_INT, ROOT, MPI_COMM_WORLD);


  //initialize the log file
  sprintf(strTemp, "%d", myRank);
  strcat(strTemp, "_scatterMat.log");
  myLogFile = fopen(strTemp, "w");
    
  fprintf(myLogFile,"\nrank:%d divide indices", myRank);

  //divide row indices among matrix
  if (myRank == ROOT) {
    fprintf(myLogFile,"\ndisplay csr rowptr b4 dividing indices: ");
    logArray(csrMat->rowPtr, csrMat->numRows+1, myRank, myLogFile);

    fprintf(myLogFile,"\nrank:%d inside divide indices", myRank);
    //divide rows ind among matrix
    totalRows = csrMat->numRows;
    rowPerProc = totalRows / numProcs;
    extraRows = totalRows % numProcs;
    fprintf(myLogFile,"\nrowPerProc = %d", rowPerProc);
    fprintf(myLogFile,"\nextraRows = %d", extraRows);
    prevEndRow = -1;
    //assuming row ind starts from 0
    for (i = 0; i < numProcs; i++) {
      fprintf(myLogFile,"\nprevEndRow = %d", prevEndRow);
      (rowInfo)[i] = prevEndRow + 1;
      (rowInfo)[i+numProcs] = (rowInfo)[i] + rowPerProc - 1;
      if (extraRows-- > 0) {
	(rowInfo)[i+numProcs] += 1;
      }
      prevEndRow = (rowInfo)[i+numProcs];
    }

    fprintf(myLogFile, "\nrowInfo: ");
    logArray(rowInfo, 2*numProcs, myRank, myLogFile);

    for (i = 0; i < numProcs; i++) {
      fprintf(myLogFile,"\nRank: %d First Row: %d Last Row: %d", i, (rowInfo)[i],
	     (rowInfo)[i+numProcs]);
    }
    
  }
  
  
  //send the row info to procs
  MPI_Bcast(rowInfo, 2*numProcs, MPI_INT, ROOT, MPI_COMM_WORLD);
  
  
  //communicate matrix values to proces
  if (myRank == ROOT) {
    //mark values and rows to be assigned to proc
    rowCount = (int *) malloc(sizeof(int)*numProcs);
    colCount = (int *) malloc(sizeof(int)*numProcs);
    
    dispRowPtr = (int *) malloc(sizeof(int)*numProcs);
    sendCountRowPtr = (int *) malloc(sizeof(int)*numProcs);
    
    dispColInd = (int *) malloc(sizeof(int)*numProcs);
    sendCountColInd = (int *) malloc(sizeof(int)*numProcs);
    
    fprintf(myLogFile,"\ndisplay csr rowptr b4 scattering: ");
    logArray(csrMat->rowPtr, csrMat->numRows+1, myRank, myLogFile);

    for (i = 0; i < numProcs; i++) {
      //start end assigned to proc i
      startRow = (rowInfo)[i];
      endRow = (rowInfo)[i+numProcs];

      //row count for proc i
      rowCount[i] = endRow - startRow + 1;
      
      //TODO: verify below indices
      //col count or non-zero values for proc i
      colCount[i] = csrMat->rowPtr[endRow+1] - csrMat->rowPtr[startRow];
      
      fprintf(myLogFile,"\ncolCount[%d] = rowPtr[%d] - rowPtr[%d] = %d - %d = %d", i,
	     endRow + 1, startRow,
	     csrMat->rowPtr[endRow+1],
	     csrMat->rowPtr[startRow],
	     csrMat->rowPtr[endRow+1] - csrMat->rowPtr[startRow]);

      //get offset and count of rows to be send to proc i
      dispRowPtr[i] = startRow;
      sendCountRowPtr[i] = rowCount[i];

      //get offset and count of values to be sent to proc i
      dispColInd[i] = csrMat->rowPtr[startRow];
      sendCountColInd[i] = colCount[i];

        //send nnz count to procs
      if (i != ROOT) {
	fprintf(myLogFile,"\nrank:%d sending to %d values  %d", myRank, i, *(colCount + i));
	MPI_Send(colCount+i, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
      } else {
	myNNZCount = colCount[ROOT];
      }
    }
    
    fprintf(myLogFile,"\nRowCount: ");
    logArray(rowCount, numProcs, myRank, myLogFile);

    fprintf(myLogFile,"ColCount: ");
    logArray(colCount, numProcs, myRank, myLogFile);

    fprintf(myLogFile,"dispRowPtr: ");
    logArray(dispRowPtr, numProcs, myRank, myLogFile);

    fprintf(myLogFile,"sendCountRowPtr: ");
    logArray(sendCountRowPtr, numProcs, myRank, myLogFile);

    fprintf(myLogFile,"dispColInd: ");
    logArray(dispColInd, numProcs, myRank, myLogFile);

    fprintf(myLogFile,"sendCountColInd: ");
    logArray(sendCountColInd, numProcs, myRank, myLogFile);
    
  } else {
    MPI_Recv(&myNNZCount, 1, MPI_INT, ROOT, 100, MPI_COMM_WORLD, &status);
  }
  

  
  fprintf(myLogFile,"\nRank: %d nnzcount:%d ", myRank, myNNZCount);
  fprintf(myLogFile,"\nrowInfo[%d] = %d rowInfo[%d] = %d ", myRank,
	  (rowInfo)[myRank], myRank+numProcs, (rowInfo)[myRank+numProcs]);

  //prepare storage for local csr matrix
  
  (*myCSRMat)->origFirstRow = (rowInfo)[myRank];
  (*myCSRMat)->origLastRow = (rowInfo)[myRank+numProcs];
  
  (*myCSRMat)->nnzCount = myNNZCount;
  (*myCSRMat)->numRows = (*myCSRMat)->origLastRow - (*myCSRMat)->origFirstRow  + 1;
  //number of columns is identical to original num columns 
  //in this case num rows is equal to num cols in original matrix
  (*myCSRMat)->numCols = (rowInfo)[(numProcs-1) + numProcs] + 1;

  (*myCSRMat)->rowPtr = (int *) malloc(sizeof(int) * (((*myCSRMat)->numRows)+1));
  memset((*myCSRMat)->rowPtr, 0, sizeof(int) * (((*myCSRMat)->numRows)+1));

  (*myCSRMat)->colInd = (int *) malloc(sizeof(int) * myNNZCount);
  memset((*myCSRMat)->colInd, 0, sizeof(int) * myNNZCount);

  (*myCSRMat)->values = (float *) malloc(sizeof(float) * myNNZCount);
  memset((*myCSRMat)->values, 0, sizeof(float) * myNNZCount);

  
  //communicate marked values to procs
  MPI_Scatterv(csrMat->rowPtr, sendCountRowPtr, dispRowPtr, MPI_INT,
	       (*myCSRMat)->rowPtr, (*myCSRMat)->numRows, MPI_INT, ROOT,
	       MPI_COMM_WORLD);
  MPI_Scatterv(csrMat->colInd, sendCountColInd, dispColInd, MPI_INT,
	       (*myCSRMat)->colInd, (*myCSRMat)->nnzCount, MPI_INT, ROOT,
	       MPI_COMM_WORLD);
  MPI_Scatterv(csrMat->values, sendCountColInd, dispColInd, MPI_FLOAT,
	       (*myCSRMat)->values, (*myCSRMat)->nnzCount, MPI_FLOAT, ROOT,
	       MPI_COMM_WORLD);
  
  (*myCSRMat)->rowPtr[(*myCSRMat)->numRows] = (*myCSRMat)->rowPtr[0] + myNNZCount;
  
  
  //free allocated mems
  if (rowCount && myRank == ROOT) {
      free(rowCount);
      rowCount = NULL;
  }

  if (colCount && myRank == ROOT) {
    free(colCount);
    colCount = NULL;
  }

  if (dispRowPtr && myRank == ROOT) {
      free(dispRowPtr);
      dispRowPtr = NULL;
  }


  if (sendCountRowPtr && myRank == ROOT) {
    free(sendCountRowPtr);
    sendCountRowPtr = NULL;
  }


  if (dispColInd && myRank == ROOT) {
    free(dispColInd);
    dispColInd = NULL;
  }


  if (sendCountColInd && myRank == ROOT) {
    free(sendCountColInd);
    sendCountColInd = NULL;
  }

  fclose(myLogFile);
  fprintf(myLogFile,"\n exiting matcomm rank: %d", myRank);
 
}


void scatterVector(float *vec, int *rowInfo, float *myVec) {

  int myRank, numProcs;
  int i;

  int *dispRowPtr, *sendCountRowPtr;
  int myRowCount;

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  dispRowPtr = (int *) malloc(sizeof(int) * numProcs);
  sendCountRowPtr = (int *) malloc(sizeof(int) * numProcs);

  if (myRank == ROOT) {

    for (i = 0; i < numProcs; i++) {
      //get offset and count of rows to be send to proc i
      dispRowPtr[i] = rowInfo[i];
      sendCountRowPtr[i] = rowInfo[i+numProcs] - rowInfo[i] + 1;
    }
  }

  myRowCount = rowInfo[myRank+numProcs] - rowInfo[myRank] + 1;

  //communicate marked values to procs
  MPI_Scatterv(vec, sendCountRowPtr, dispRowPtr, MPI_FLOAT,
	       myVec, myRowCount, MPI_FLOAT, ROOT,
	       MPI_COMM_WORLD);

}
