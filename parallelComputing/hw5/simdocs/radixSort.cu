#include <stdio.h>

__device__ void radixSort(float *d_InArr, float *d_OutArr, int n, int numBits) {
  
  int i;

  //get current block number
  int blockId = blockIdx.x;
  
  //thread id within a block
  int threadId = threadIdx.x;

  //global thread id
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  //TODO: shared mem allocatiom
  //shared memory to store histogram count 
  //size (2**numBits)
  __shared__ int sh_count[];
  
  //TODO: shared mem allocation
  //size is of num threads in block
  __shared__ int sh_tempArr[];
  __shared__ int sh_tempPred[];

  //copy input to tempArr
  //assuming total number of threads greater than number of elems
  if (globalThreadId < n) {
    tempArr[threadId] = d_InArr[globalThreadId];
  }

  //initialize shared count to zero
  sh_tempArr[threadId] = 0;

  
}



__global__ void onChipPreSort(int *d_inArr, int n,
			      int startBit, int numBits) {
  int i;

  //get current block number
  int blockId = blockIdx.x;
  
  //thread id within a block
  int threadId = threadIdx.x;

  //global thread id
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  //block's array to sort
  //TODO:
  __shared__ int sh_blockArr[];
  __shared__ int sh_blockPred[];
  __shared__ int sh_blockOut[];

  //won't this cause thread divergence
  if (globalThreadId < n) {
    sh_blockArr[threadId] = d_inArr[globalThreadId];
    for (i = 0; i < numBits; i++) {
      //TODO: BLOCKSIZE
      sortPerBit(startBit+i, BLOCKSIZE, sh_blockArr,
		 sh_blockPred, sh_blockOut);
      sh_blockArr[threadId] = sh_blockOut[threadId];
    }
    
  }

  //sh_blockOut is having final sorted output
  
  
}



//bitPos starts from 0, n -> block element count
__device__ void sortPerBit(int bitPos, int n,
		     int *sh_tempArr, int *sh_tempPred,
		     int *sh_tempOut) {
  
  int i, key, totalFalses;

  __shared__ int lastPred;
  __shared__ int t[];
  __shared__ int d[];

  //get current block number
  int blockId = blockIdx.x;
  
  //thread id within a block
  int threadId = threadIdx.x;

  //global thread id
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  //set 1 for 0 at bitPos
  if (threadId < n) {
    sh_tempPred[threadId] = ((sh_tempArr[threadId]>>bitPos) & 1) == 0;
  }
  
  if (threadId == n-1) {
    //last thread for last element
    lastPred = sh_tempPred[threadId];
  }

  //scan the 1's
  //TODO: set n
  preScan(sh_tempArr, sh_tempPred, n);

  totalFalses = sh_tempPred[n-1] + lastPred;

  //t = i - f + totalFalses
  if (threadId < n) {
    t[threadId] = threadId - sh_tempPred[threadId] + totalFalses;
    d[threadId] = ((sh_tempArr[threadId]>>bitPos) & 1) ? t[threadId] : sh_tempPred[threadId];
  }
  
  //scater input using d as scatter address
  if (threadId < n) {
    sh_tempOut[d[threadId]] =  sh_tempArr[threadId];
  }
}



__device__ void preScan(int *arr, int *arrPred, int n) {
  int ai, bi;
  int thId = threadIdx.x;
  int d = 0, offset = 1;
  int temp;

  //build sum in place
  for (d = n>>1; d > 0; d >>=1) {
    __syncthreads();
    if (thId < d) {
      ai = offset*(2*thId+1) - 1;
      bi = offset*(2*thId+2) - 1;
      arrPred[bi] += arrPred[ai];
    }
    offset*=2;
  }
  
  //clear last element
  if (thId == 0) {
    arrPred[n-1] = 0;
  }

  //traverse down tree & build scan
  for (d = 1; d < n; d *=2) {
    offset >> = 1;
    __syncthreads();
    if (thId < d) {
      ai = offset*(2*thId + 1) - 1;
      bi = offset*(2*thId + 2) - 1;
      temp = arrPred[ai];
      arrPred[ai] = arrPred[bi];
      arrPred[bi] += temp;
    }
  }

  __syncthreads();
  
}






__device__ void sortStep(int stepNum, int numBits, int n,
			 int *sh_tempArr, int *sh_count) {
  int i, key;
  
  //get current block number
  int blockId = blockIdx.x;
  
  //thread id within a block
  int threadId = threadIdx.x;

  //global thread id
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  int mask = (1<<numBits) - 1;
 
  //get histogram of keys
  if (globalThreadId < n) {
    key = (sh_tempArr[threadId] >> stepNum*numBits) & mask;
    atomicAdd(&(sh_count[key]), 1);
  }

  //



}
