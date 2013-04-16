
#include "vecComm.h"
#inlcude "mpi.h"
#include <stdlib.h>


void addToSet(int *set, int val) {
  int block, pos;
  
  block = val/sizeof(int);
  pos = val%sizeof(int);

  set[block] = set[block] | (1 << pos);
}


int isInSet(int *set, int val) {
  int block, pos;
  
  block = val/sizeof(int);
  pos = val%sizeof(int);

  return set[block] & (1 << pos);
}


int sizeSet(int *set, int capacity) {
  int i, j, size;
  size = 0;
  for (i = 0; i < capacity; i++) {
    for (j = 0; j < sizeof(int); j++) {
      if ((1 << j) & set[i])
	size++;
    }
  }
  return size;
}


int* getSetElements(int *set, int capacity) {
  int i, j, size;
  int *elem, k;
  size = sizeSet(int *set, int capacity);
  elem = malloc(sizeof(int)* size);
  k = 0;
  for (i = 0; i < capacity; i++) {
    for (j = 0; j < sizeof(int); j++) {
      if ((1 << j) & set[i])
	elem[k++] = i*sizeof(int) + j;
    }
  }
  return elem;
}


//will prepare bVecParams to store the required vector elements by the 
//current process
void prepareVectorComm(CSRMat* myCSRMat, BVecComParams *bVecParams,
		       int *rowInfo) {
  int myRank, numProcs;
  int i, j, setCapacity;
  int* bitColSet, recvIdx; 
  int recvCount;
  int *modRowInfo;
  int *tempPtr, *tempProc;

  bitColSet = (int *)0;
  recvCol = (int *)0;
  tempProc = (int *)0;
  tempPtr = (int *)0;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  
  //modRowInfo[2*rank]-start row modRowInfo[2*rank+1]- end row
  modRowInfo = (int *) malloc(sizeof(int)*2*numProcs);
  for (i = 0; i < numProcs; i++) {
    modRowInfo[i*2] = rowInfo[i];
    modRowInfo[i*2+1] = rowInfo[i+numProcs]
  }

  //determine which index of vector needed by process by scanning column indices
  //use bitColSet and those 
  //TODO: check (int)0.4 or 0.5 roundoff 
  setCapacity = ((myCSRMat->numCols+1)/sizeof(int)*8) + 0.5;
  bitColSet = (int*) calloc(setCapacity, sizeof(int));
  for (i = 0; i < myCSRMat->nnzCount; i++) {
    if (myCSRMat->colInd[i] >= myCSRMat->origFirstRow &&
	myCSRMat->colInd[i] <= myCSRMat->origLastRow) {
      //current proc has these columns, no need to add these to set
      continue;
    }
    addToSet(myCSRMat->colInd[i]);
  }
  
  //get the count of columns to receive from remote proc
  recvCount = sizeSet(bitColSet, setCapacity);
  
  //get vectr idx to receive
  recvIdx = getSetElements(bitColSet, setCapacity);

  //get the remote process that contain these vector elements
  tempPtr = (int *) malloc(sieof(int)*numProcs);
  tempProc = (int *) malloc(sieof(int)*numProcs);
  for (i = 0; i < numProcs; i++) {
    tempPtr[i] = -1;
    tempProc[i] = -1;
  }
  
  j = -1;
  for (i = 0; i < recvCount; i++) {
    //search recvIdx[i] in modRowInfo, say get proc k
    //TODO:
    k = -1;
    if (j == -1 || tempProc[j] != k) {
      //found a new rank, add it to list of proc to recv from
      tempProc[++j] = k;
      //set starting index of this proc
      tempPtr[j] = i;
    }
  }

  //set the above information about receiving in bVecParams
  bVecParams->numToRecvProcs = j+1;

  //set the procs from which elements of vector will be received
  bVecParams->toRecvProcs = (int *) malloc(sizeof(int) *
					   bVecParams->numToRecvProcs);
  memcpy(bVecParams->toRecvProcs, tempProc, bVecParams->numToRecvProcs);

  //set the pointer to elements that will be received from proc
  bVecParams->recvPtr = (int *) malloc(sizeof(int) *
				       bVecParams->numToRecvProcs+1);
  memcpy(bVecParams->recvPtr, tempPtr, bVecParams->numToRecvProcs);
  //set last elements of recvptr for last proc
  bVecParams->recvPtr[bVecParams->numToRecvProcs] = recvCount;




}

