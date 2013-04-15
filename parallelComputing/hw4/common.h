/*
 * contains helper struct and methods used across procs
 */

#ifndef _COMMON_
#define _COMMON_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROOT 0

//format of sparse matrix used i.e. Compressed Sparse Row format
typedef struct {
  //store where and which elements of row are stored
  int *rowPtr;

  //column indices of row
  //for row i, column indices are stored starting at colInd[rowPtr[i]] &
  //ending at colInd[rowPtr[i+1]] (not included)
  int *colInd;

  //values of row i, values[rowPtr[i]] to values[rowPtr[i+1]] (not included)
  float *values;

  //num of rows hold by the proc
  int numRows;
  //num of cols hold by the procs
  int numCols;
  //number of non-zero values or size of values[]
  int nnzCount;
} CSRMat;


//store required parameters for communication of b vector
typedef struct {
  //num of procs to send
  int numToSendProcs;
  //ranks of procs to send
  int *toSendProcs;
  
  //num of procs to receive from
  int numToRecvProcs;
  //ranks of procs to receive from
  int *toRecvProcs;
  
  //store the indices of elements of b in sendInd to send to other procs, i.e.
  //send to ith neighbor elements b/w: sendInd[sendPtr[i]]
  //and sendInd[sendPtr[i+1] - 1] (including both locations). 
  //size: numToSendProcs+1
  int *sendPtr;
  
  //elements of b vector to send to other procs, size equal to no. of elements
  //required to send all other procs
  int *sendInd;

  
  //store the indices of elements of b in recvInd to recv from other procs, i.e.
  //recv from ith neighbor elements b/w: recvInd[recvPtr[i]]
  //and recvInd[recvPtr[i+1] - 1] (including both locations). 
  //size: numToRecvProcs+1
  int *recvPtr;
  
  //elements of b vector to recv from other procs, size equal to no. of elements
  //required to recv from other procs
  int *recvInd;
  
  //store elements of b that are sent
  int *sendBuf;

  //store elements of b that are received
  int *recvBuf;
  
} BVecComParams;


#endif
