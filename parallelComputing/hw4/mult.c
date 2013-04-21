#include "mult.h"
#include <stdio.h>
#include <string.h>

#include "io.h"
# include "debug.h"
/*compute the product of local matrix and vector*/

void computeLocalProd(CSRMat *myCSRMat, BVecComParams *myVecParams,
		      float *localVec, float *locProdvec, int myRank) {
  
  int i, j;
  int startInd, endInd;
  int row, col;
  float matVal, vecVal;
  int searchedInd;
  char strTemp[20];
  FILE *myLogFile;
  
  myLogFile = NULL;
  
  if (DEBUG) {
    //initialize the log file
    sprintf(strTemp, "%d", myRank);
    strcat(strTemp, "_locProd.log");
    myLogFile = fopen(strTemp, "w");
    
    dbgPrintf(myLogFile, "\n numRows = %d", myCSRMat->numRows);
    dbgPrintf(myLogFile, "\n recvInd: ");
    logArray(myVecParams->recvInd, myVecParams->recvCount, myRank, myLogFile);
    dbgPrintf(myLogFile, "\n recvBuf: ");
    logFArray(myVecParams->recvBuf, myVecParams->recvCount, myRank, myLogFile);
  }

  for (row = 0; row < myCSRMat->numRows; row++) {

    dbgPrintf(myLogFile, "\nrow = %d ", row);

    startInd = myCSRMat->rowPtr[row] - myCSRMat->rowPtr[0];
    endInd = myCSRMat->rowPtr[row+1] - myCSRMat->rowPtr[0];

    locProdvec[row] = 0;

    for (j = startInd; j < endInd; j++) {
      col = myCSRMat->colInd[j];
      matVal = myCSRMat->values[j];
      
      dbgPrintf(myLogFile, "\nrow = %d col = %d ", row, col);

      if (myCSRMat->origFirstRow <= col && col <= myCSRMat->origLastRow) {
	//column needed is present in local processor use it
	vecVal = localVec[col - myCSRMat->origFirstRow];
	dbgPrintf(myLogFile, "\n column:%d found locally at %d val=%f", col,
		col - myCSRMat->origFirstRow, vecVal);
      } else {
	//else search for column value in received column indices
	//TODO: think of something better than binary search
	searchedInd = binIndSearch(myVecParams->recvInd, myVecParams->recvCount, col);
	dbgPrintf(myLogFile, "\nsearch for column: %d search res: %d", col, searchedInd);
	if (searchedInd == -1) {
	  dbgPrintf(myLogFile, "\ncolumn not found, this shouldn't happen");
	} else {
	  vecVal = myVecParams->recvBuf[searchedInd];
	}
      }

      dbgPrintf(myLogFile, " matval = %f val = %f", matVal, vecVal);

      locProdvec[row] += vecVal * matVal;
    }
    
  }

  if (NULL != myLogFile) {
    fclose(myLogFile);
  }
}


//compute the product of full matrix with the full vector
void computeSerialProd(CSRMat *csrMat, float *fullVec, float *prodVec) {
  int i, j;
  int startInd, endInd;
  int row, col;
  float matVal, vecVal;
  int searchedInd;

  //printf("\n serial prod: numRows=%d", csrMat->numRows );

  for (row = 0; row < csrMat->numRows; row++) {
    startInd = (csrMat->rowPtr)[row];
    endInd = (csrMat->rowPtr)[row+1];
    //printf("\n row: %d startInd: %d endInd: %d", row, startInd, endInd);
    prodVec[row] = 0;

    for (j = startInd; j < endInd; j++) {
      col = (csrMat->colInd)[j];
      matVal = (csrMat->values)[j];
      vecVal = fullVec[col];
      prodVec[row] += vecVal * matVal;
    }

  }

}
