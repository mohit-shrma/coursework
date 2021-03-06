/*
 * contains helper struct and methods used across procs
 */

#ifndef _COMMON_
#define _COMMON_

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

  //store mapping to original mat of row
  int origFirstRow, origLastRow;
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

  //num of values sent
  int sendCount;

  //num of values received
  int recvCount;
  
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
  float *sendBuf;

  //store elements of b that are received
  float *recvBuf;
  
} BVecComParams;

void init(BVecComParams *bVecParams);

void freeBVecComParams(BVecComParams *bVecParams);
void initCSRMat(CSRMat *csrMat);
void freeCSRMat(CSRMat *csrMat);
int binIndSearch(int *arr, int len , int val);

double getTime();

#endif
