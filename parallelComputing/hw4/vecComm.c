#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "vecComm.h"
#include "io.h"

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
  size = sizeSet(set, capacity);
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


//search for k in array arr and return the start of interval it lies in
int modBinSearch(int *arr, int len, int k) {
  int lb, ub, mid;
  
  lb = 0, ub = len-1;

  while (lb < ub) {
    mid = (lb+ub)/2;
    if (k == arr[mid]) {
      //found the exact val, passed row is either start or end
      if (mid %2 == 0) {
	//start row
	return mid;
      } else {
	//end row
	return mid-1;
      }
    } else {
      if (arr[mid] < k && k < arr[mid+1]) {
	if (mid % 2 == 0) {
	  return mid;
	} else {
	  printf("\n shouldn't reach here %d < %d < %d", arr[mid], k,
		 arr[mid+1]);
	  return -1;
	}
      } else if (arr[mid-1] < k && k < arr[mid]) {
	if ((mid - 1) %2 == 0) {
	  return mid -1;
	} else {
	  printf("\n shouldn't reach here %d < %d < %d", arr[mid-1], k,
		 arr[mid]);
	  return -1;
	}
      } else if (k > arr[mid]) {
	lb = mid;
      } else if (k < arr[mid]) {
	ub = mid;
      }
    }
  }
  return -1;
}



//will prepare bVecParams to store the required vector elements by the 
//current process
void prepareVectorComm(CSRMat* myCSRMat, float *myVec,
		       BVecComParams *bVecParams, int *rowInfo) {
  int myRank, numProcs;
  int i, j, k, setCapacity;
  int *bitColSet, *recvIdx; 
  int recvCount;
  int *modRowInfo;
  int *tempPtr, *tempProc;
  int *sends, *receives;
  int sendCount;

  MPI_Request *sendRequest, *recvRequest;

  bitColSet = (int *)0;
  recvIdx = (int *)0;
  modRowInfo = (int *)0;
  tempProc = (int *)0;
  tempPtr = (int *)0;
  sends = (int *)0;
  receives = (int *)0;
  sendRequest = (MPI_Request *) 0;

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


  //modRowInfo[2*rank]-start row modRowInfo[2*rank+1]- end row
  modRowInfo = (int *) malloc(sizeof(int)*2*numProcs);
  for (i = 0; i < numProcs; i++) {
    modRowInfo[i*2] = rowInfo[i];
    modRowInfo[i*2+1] = rowInfo[i+numProcs];
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
    addToSet(bitColSet, myCSRMat->colInd[i]);
  }
  
  //get the count of columns to receive from remote proc
  recvCount = sizeSet(bitColSet, setCapacity);
  printf("rank: %d recvCount=%d", myRank, recvCount);
  
  //get vectr indices that we want to receive
  recvIdx = getSetElements(bitColSet, setCapacity);
  printf("recvIdx: ");
  dispArray(recvIdx, recvCount, myRank);
  
  /*
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
    k = modBinSearch(modRowInfo, 2*numProcs, recvIdx[i]);

    if (k == -1) {
      printf("\n Couldn't find rank for %d", recvIdx[i]);
    }

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
  printf(" toRecvProcs: ");
  dispArray(bVecParams->toRecvProcs, bVecParams->numToRecvProcs, myRank);

  //set the pointer to elements that will be received from proc
  bVecParams->recvPtr = (int *) malloc(sizeof(int) *
				       bVecParams->numToRecvProcs+1);
  memcpy(bVecParams->recvPtr, tempPtr, bVecParams->numToRecvProcs);
  //set last elements of recvptr for last proc
  bVecParams->recvPtr[bVecParams->numToRecvProcs] = recvCount;

  //initialize recvInd and recvBuf
  bVecParams->recvInd = (int *) malloc(sizeof(int) * recvCount);
  bVecParams->recvBuf = (int *) malloc(sizeof(int) * recvCount);

  //initialize recvInd with columns/vector indices that proc needs
  memcpy(bVecParams->recvInd, recvIdx, recvCount);
  printf(" recvInd: ");
  dispArray(bVecParams->recvInd, recvCount, myRank);
  
  /*
  //allocate memory for send and receive buffer
  sends = (int*) malloc(sizeof(int)*numProcs);
  memset(sends, 0, numProcs);

  receives = (int*) malloc(sizeof(int)*numProcs);
  memset(receives, 0, numProcs);

  //initialize receives buffer with count of elements
  //it needs to receive from process
  for (i = 0; i < bVecParams->numToRecvProcs; i++) {
    receives[bVecParams->toRecvProcs[i]] =
      bVecParams->recvPtr[bVecParams->toRecvProcs[i+1]] -
      bVecParams->recvPtr[bVecParams->toRecvProcs[i]]; 
  }


  //ALL-TO-ALL comm to let each processor know where to send 
  MPI_Alltoall(receives, 1, MPI_INT, sends, 1, MPI_INT, MPI_COMM_WORLD);

  //now sends contain information how many elements we need to
  //send to each process 

  //get the number of processes we need to send info
  for (i = 0, j = 0; i < numProcs; i++) {
    if (receives[i] != 0) {
      j++;
    }
  }

  //store the number of processes to send
  bVecParams->numToSendProcs = j;
  
  bVecParams->toSendProcs = (int *) malloc(sizeof(int) *
					   bVecParams->numToSendProcs);
  
  //number of elements to send 
  sendCount = 0

  //set the count of elements to send to processes
  bVecParams->sendPtr = (int *) malloc(sizeof(int) *
				       bVecParams->numToSendProcs+ 1);
  bVecParams->sendPtr[j] = 0;
  for (i = 0, j = 1, k = 0; i < numProcs; i++) {
    if (receives[i] != 0) {
      bVecParams->sendPtr[j] = receives[i] + bVecParams->sendPtr[j];
      sendCount += receives[i];
      bVecParams->toSendProcs[k++] = i;
      j++;
    }
  }
  printf("sendPtr: ");
  dispArray(sendPtr, bVecParams->numToSendProcs+ 1, myRank);
  
  
  //allocate space for sendInd & sendBuf in bVecParams
  bVecParams->sendInd = (int *) malloc(sizeof(int)*sendCount);
  bVecParams->sendBuf = (int *) malloc(sizeof(int)*sendCount);

  //at this point every processor knows what it needs to receive but not what
  //needs to send

  //perform a non blocking send of indices/columns it needs to receive
  
  //initialize MPI_Request
  sendRequest = (MPI_Request*) malloc(sizeof(MPI_Request)
				      * bVecParams->numToRecvProcs);
  for (i = 0; i < bVecParams->numToRecvProcs; i++) {
    MPI_Isend(bVecParams->recvInd + bVecParams->recvPtr[i] ,
	      bVecParams->recvPtr[i+1] - bVecParams->recvPtr[i],
	      MPI_INT, bVecParams->toRecvProcs[i], 100, MPI_COMM_WORLD,
	      sendRequest + i);
  }

  //perform a non-blocking receive of columns/indices needed by another process
  //initialize receive MPI_Request
  recvRequest = (MPI_Request*) malloc(sizeof(MPI_Request)
				      * bVecParams->numToSendProcs);
  for (i = 0; i < bVecParams->numToSendProcs; i++) {
    MPI_Irecv(bVecParams->sendInd + bVecParams->sendPtr[i] ,
	      bVecParams->sendPtr[i+1] - bVecParams->sendPtr[i],
	      MPI_INT, bVecParams->toSendProcs[i], 100, MPI_COMM_WORLD,
	      recvRequest + i);
  }

  //TODO: avoid below barrier or check for requests above
  MPI_Barrier();

  printf("sendInd :");
  dispArray(sendInd, sendCount, myRank);

  //at this point each processor know what it needs to send in bVecParams->sendInd

  //copy the elements it needs to send in bVecParams->sendBuf
  for (i = 0, k = 0; i < bVecParams->numToSendProcs; i++) {
    for (j = bVecParams->sendPtr[i]; j < bVecParams->sendPtr[i+1]; j++) {
      bVecParams->sendBuf[k++] = myVec[bVecParams->sendInd[j]];
    }
  }

  printf("sendBuf: ");
  dispArray(sendBuf, sendCount, myRank);

  //we know what we need to receive from other processors, issue non-blocking
  //receive operations for these
  for (i = 0; i < bVecParams->numToRecvProcs; i++) {
    MPI_Irecv(bVecParams->recvBuf + bVecParams->recvPtr[i] ,
	      bVecParams->recvPtr[i+1] - bVecParams->recvPtr[i],
	      MPI_INT, bVecParams->toRecvProcs[i], 100, MPI_COMM_WORLD,
	      sendRequest + i);
  }

  //send appropriate local elements of b vector
  for (i = 0; i < numToSendProcs; i++) {
    MPI_Send(bVecParams->sendBuf + bVecParams->sendPtr[i],
	     bVecParams->sendPtr[i+1] - bVecParams->sendPtr[i],
	     MPI_INT, bVecParams->toSendProcs[i], 100, MPI_COMM_WORLD,);
  }
  */
  
  

  if (sends) {
    free(sends);
  }

  if (receives) {
    free(receives);
  }

}

