#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io.h"

//split the passed line into row, col, value
void splitMatRow(char *line, int *row, int *col, float *val) {
  int i, j;
  char *temp;
  
  *row = -1;
  *col = -1;
  *val = -1;
    
  temp = malloc(BUF_SZ);
  for (i=0, j=0; i < BUF_SZ && line[i] != '\0'; i++, j++) {
    if (line[i] != '\t' && line[i] != ' ' && line[i] != '\n') {
      temp[j] = line[i];
    } else {
      temp[j] = '\0';
      j=-1;
      if (*row == -1) {
	*row = atoi(temp);
      } else if (*col == -1) {
	*col = atoi(temp);
      } else {
	*val = atof(temp);
	break;
      }
    }
  }

  free(temp);
}


//count the number of lines to know the dimension of matrix and vector
void getDimNCount(char *matFileName, int *dim, int *nnz) {
  int nnzCount, ch;
  FILE *fin;
  char *line, *dupLine;

  int row, col;
  float val;

  nnzCount = 0;
  line = malloc(BUF_SZ);
  dupLine = malloc(BUF_SZ);


  if ((fin = fopen(matFileName, "r")) == NULL) {
    fprintf(stderr, \
	    "Error: failed to read file %s while calculating dimensions\n",\
	    matFileName);
    exit(1);
  } else {

    while ((fgets(line, BUF_SZ, fin)) != NULL) {
      nnzCount++;
      memcpy(dupLine, line, BUF_SZ);      
    }

    //get the last row ind
    splitMatRow(dupLine, &row, &col, &val);
    *nnz = nnzCount;
    *dim = row+1;
  }

  
  fclose(fin);
  free(line);
  free(dupLine);
}




//display sparse matrix
void displSparseMat(CSRMat *csrMat, int rank) {
  int i, j;
  
  int startValInd, endValInd;

  printf("\nmyCSRMat Rank:%d numRows:%d", rank, csrMat->numRows);
  printf("\nmyCSRMat Rank:%d nnz count:%d", rank, csrMat->nnzCount);
  printf("\nmyCSRMat Rank: %d rowPtr: ", rank);
  dispArray(csrMat->rowPtr, csrMat->numRows+1, rank);
  
  for (i = 0; i < csrMat->numRows; i++) {
    startValInd = csrMat->rowPtr[i] - csrMat->rowPtr[0];
    endValInd = csrMat->rowPtr[i+1] - csrMat->rowPtr[0];
    for (j = startValInd; j < endValInd; j++) {
      //printf("\nrank=%d\t%d\t%d\t%f\t%d", rank, i, csrMat->colInd[j], csrMat->values[j], j);
    }
  }
  printf("\n");
}


void logSparseMat(CSRMat *csrMat, int rank, FILE* myLogFile) {
  int i, j;
  
  int startValInd, endValInd;

  fprintf(myLogFile, "\nmyCSRMat Rank:%d numRows:%d", rank, csrMat->numRows);
  fprintf(myLogFile, "\nmyCSRMat Rank:%d nnz count:%d", rank, csrMat->nnzCount);
  fprintf(myLogFile, "\nmyCSRMat Rank: %d rowPtr: ", rank);
  logArray(csrMat->rowPtr, csrMat->numRows+1, rank, myLogFile);
  
  for (i = 0; i < csrMat->numRows; i++) {
    startValInd = csrMat->rowPtr[i] - csrMat->rowPtr[0];
    endValInd = csrMat->rowPtr[i+1] - csrMat->rowPtr[0];
    for (j = startValInd; j < endValInd; j++) {
      //fprintf(myLogFile, "\nrank=%d\t%d\t%d\t%f\t%d", rank, i, csrMat->colInd[j], csrMat->values[j], j);
    }
  }
  fprintf(myLogFile, "\n");
}


//read the sparse matrix dim X dim and store it in CSR format 
CSRMat* readSparseMat(char *matFileName, int dim, int nnz) {
 
  CSRMat *csrMat;
  FILE *matFile;
  char *line;
  int i, row, col;
  float val;
  int prevRow;

  csrMat = (CSRMat *) 0;
  line = (char *)0;
  prevRow = -1;

  if ((matFile = fopen(matFileName, "r")) == NULL) {
    fprintf(stderr, "Error: failed to read file %s \n", matFileName);
    exit(1);
  } else {

    //assign dimensions and size of matrix
    csrMat = (CSRMat *) malloc(sizeof(CSRMat));

    csrMat->nnzCount = nnz;
    csrMat->numRows = dim;
    csrMat->numCols = dim;
    
    csrMat->rowPtr = (int *) malloc(sizeof(int) * dim);
    csrMat->colInd = (int *) malloc(sizeof(int) * nnz);
    csrMat->values = (float *) malloc(sizeof(float) * nnz);
    
    csrMat->origFirstRow = 0;
    csrMat->origLastRow = dim -1;

    //read matrix file to fill the matrix
    line = malloc(BUF_SZ);

    for (i = 0; i < nnz; i++) {
      fgets(line, BUF_SZ, matFile);
      splitMatRow(line, &row, &col, &val);
      
      csrMat->colInd[i] = col;
      csrMat->values[i] = val;
      
      if (row != prevRow) {
	//new row begins
	csrMat->rowPtr[row] = i;
	prevRow = row;
      }
    }

    //set the last val in rowPtr
    csrMat->rowPtr[prevRow+1] = nnz;
  }

  if (line) {
    free(line);
  }

  fclose(matFile);

  return csrMat;
}


//read the sparse vector of size dim and return
void readSparseVec(float *bVec, char* vecFileName, int dim) {

  FILE *vecFile;
  char *line;
  int i;

  line = (char *)0;

  if ((vecFile = fopen(vecFileName, "r")) == NULL) {
    fprintf(stderr, "Error: failed to read file %s \n", vecFileName);
  } else {
    line = malloc(BUF_SZ);
    for (i = 0; i < dim; i++) {
      fgets(line, BUF_SZ, vecFile);
      bVec[i] = atof(line);
    }
  }

  if (line) {
    free(line);
  }
  fclose(vecFile);
}



void logArray(int *arr, int len, int rank, FILE *logFile) {
  int i;
  fprintf(logFile, " rank:%d ", rank);
  for (i = 0; i < len; i++) {
    fprintf(logFile, "%d ", arr[i]);
  }
  fprintf(logFile,"\n");
}


void logFArray(float *arr, int len, int rank, FILE *logFile) {
  int i;
  for (i = 0; i < len; i++) {
    fprintf(logFile, "\nrank=%d arr[%d]=%f ", rank, i,arr[i]);
  }
  fprintf(logFile, "\n");
}


void dispArray(int *arr, int len, int rank) {
  int i;
  printf(" rank:%d ", rank);
  for (i = 0; i < len; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}


void dispFArray(float *arr, int len, int rank) {
  int i;
  for (i = 0; i < len; i++) {
    printf("\nrank=%d arr[%d]=%f ", rank, i,arr[i]);
  }
  printf("\n");
}

