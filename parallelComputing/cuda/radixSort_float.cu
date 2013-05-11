#include <stdio.h>
#include <cuda_runtime.h>


#define DEBUG 0

//will compute local histogram
//assuming passed pointers are adjusted for the thread
//bitpos is the lsb from which to consider numbits towards msb
__device__ void computeLocalHisto(int *localHisto, float *arrElem, int n,
				  int numBits, int bitpos) {
  
  int i;
  int numBuckets = 1 << numBits;
  int mask = (1 << numBits) - 1;
  int key;

  for (i = 0; i < n; i++) {
    key = (((int)arrElem[i]) >> bitpos) & mask;
    localHisto[key]++;
  }
  
}



__device__ void dispArr(int *arr, int n) {
  
  int i;

  //threadId with in a block, DMat doc to start with
  int thId = threadIdx.x; 
  

  if (thId == 0) {
    printf("\n");
    for (i = 0; i < n; i++) {
      printf(" %d ", arr[i]);
    }
    printf("\n");
  }
}


//assuming sizeof int == sizeof float
__device__ void computeAtomicHisto(int *aggHisto, float *arrElem, int numElem,
				   int numBits, int bitpos) {
  
  int i, j;
  int numBuckets = 1 << numBits;
  int mask = (1 << numBits) - 1;
  int key;
  void *vptr;
  int *iptr;
  //thread id within a block
  int threadId = threadIdx.x;
  
  //number of threads in block
  int nThreads = blockDim.x;

  for (i = threadId; i < numElem; i+=nThreads) {
    vptr = (void*)(arrElem + i);
    iptr = (int*)vptr;
    key =   ( (*iptr) >> bitpos)  & mask;
    atomicAdd(&(aggHisto[key]), 1);
  }

}

//assuming sizeof int == sizeof float
__device__ void writeSortedVals(int *aggHisto, float *fromArr, float *toArr,
				int numBits, int bitpos, int n) {
  int i, key;
  int mask = (1 << numBits) - 1;
  void *vptr;
  int *iptr;

  for (i = 0; i < n; i++) {
    vptr = (void*)(fromArr + i);
    iptr = (int*)vptr;

    key = (  (*iptr) >> bitpos) & mask;

    if (DEBUG) {
      printf("toArr[%d] = %f\n", aggHisto[key], fromArr[i]);
    }

    toArr[aggHisto[key]++] = fromArr[i];
  }
}


__device__ void zeroedInt(int *arr, int count) {
  int i;
  
  //thread id within a block
  int threadId = threadIdx.x;
  
  //number of threads in block
  int nThreads = blockDim.x;

  for (i = threadId; i < count; i+=nThreads) {
    arr[i] = 0;
  }
}



//scan array arr of size n=nThreads, power of 2
__device__ void preSubScan(int *arr, int n, int prev) {

  int i, d, ai, bi, offset, temp;
  //threadId with in a block, DMat doc to start with
  int thId = threadIdx.x; 

  //number of threads in blocks
  int nThreads = blockDim.x;

  d = 0;
  offset = 1;

  //build sum in place up the tree
  for (d = n>>1; d > 0; d >>=1) {
    __syncthreads();
    if (thId < d) {
      ai = offset*(2*thId+1) - 1;
      bi = offset*(2*thId+2) - 1;
      arr[bi] += arr[ai];
    }
    offset*=2;
  }
  
  //clear last element
  if (thId == 0) {
    arr[n-1] = 0;
  }

  //traverse down tree & build scan
  for (int d = 1; d < n; d *=2) {
    offset = offset >> 1;
    __syncthreads();
    if (thId < d) {
      ai = offset*(2*thId + 1) - 1;
      bi = offset*(2*thId + 2) - 1;
      temp = arr[ai];
      arr[ai] = arr[bi];
      arr[bi] += temp;
    }
  }

  for (i = thId; i < n; i+=nThreads) {
    arr[i] += prev;
  }

  __syncthreads();
}



__device__ void d_dispFArr(float *arr, int n) {
  int i;

  //threadId with in a block, DMat doc to start with
  int thId = threadIdx.x; 
  
  if (thId == 0) {
    printf("\n");
    for (i = 0; i < n; i++) {
      printf(" %f ", arr[i]);
    }
    printf("\n");
  }

}


//works efficiently for power of 2
__device__ void scan(int *arr, int n) {
  
  int i, j, prev, next, temp;

  //threadId with in a block, DMat doc to start with
  int thId = threadIdx.x; 

  //number of threads in blocks
  int nThreads = blockDim.x;


  //divide the simpred into nThreads blocks,
  //scan each block in parallel, with next iteration using results from prev blocks
  prev = 0;
  next = 0;

  for (i = 0; i < n; i += nThreads) {
    //dispArr(arr, n);
    next = arr[i+nThreads-1];
    if (n - i >= nThreads) {
      preSubScan(arr + i, nThreads, (i>0?arr[i-1]:0) + prev);
    } else {
      //not power of 2 perform serial scan for others
      //this will be last iteration of loop
      if (thId == 0) {
	for (j = i; j < n; j++) {
	  temp = prev + arr[j-1];
	  prev = arr[j];
	  arr[j] = temp;
	}
      }      
    }//end else
    
    prev = next;

  }//end for

  __syncthreads();
} 




//numbits means bits at a time
__global__ void radixSort(float *d_InArr, int n, int numBits) {
  
  int i, j, elemPerThread;
  int localHistoElemCount;

  //get current block number
  int blockId = blockIdx.x;
  
  //thread id within a block
  int threadId = threadIdx.x;

  //number of threads in block
  int nThreads = blockDim.x;

  //global thread id
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ int s[];

  //shared mem space for aggregated histogram
  int *aggHisto = s;

  //shared mem space to copy array to be sorted
  float *fromArr = (float*) &aggHisto[1<<numBits];
  float *toArr = (float *) &fromArr[n];
  float *tempSwap;

  //bucket size
  int bucketSize = 1 << numBits;

  //initialize arrays in shared mem
  for (i = threadId; i < n; i+=nThreads) {
    fromArr[i] = d_InArr[i];
    toArr[i] = 0;
  }
  
  if (threadId == 0 && DEBUG) {
    printf("\n fromArray:  ");
    d_dispFArr(fromArr, n);
  }


  //for each numbits chunk do following
  for (i = 0; i < sizeof(float)*8; i+=numBits) {
    //reset histogram
    zeroedInt(aggHisto, bucketSize);

    if (threadId == 0 && DEBUG) {
      printf("\n fromArray b4 histo :  ");
      d_dispFArr(fromArr, n);
    }

    //aggregate in histogram in shared mem
    computeAtomicHisto(aggHisto, fromArr, n,
		       numBits, i);

    if (threadId == 0 && DEBUG) {
      printf("\naggHisto, bitpos:%d:", i);
      dispArr(aggHisto, bucketSize);
      printf("\n fromArray after histo :  ");
      d_dispFArr(fromArr, n);
    }
    
    //perform scan on aggHisto (assuming power of 2)
    scan(aggHisto, bucketSize);

    if (threadId == 0 && DEBUG) {
      printf("\naggHisto after scan, bitpos:%d:", i);
      dispArr(aggHisto, bucketSize);
    }

    __syncthreads();

    if (threadId == 0) {
      //copy values to correct output by a single thread
      writeSortedVals(aggHisto, fromArr, toArr,
		      numBits, i, n);

    }
    __syncthreads();

    if (threadId == 0 && DEBUG) {
      printf("\n sorted:  ");
      d_dispFArr(toArr, n);
    }

    //toArr contains the sorted arr, for the next iteration point fromArr to this location
    tempSwap = toArr;
    toArr = fromArr;
    fromArr = tempSwap;  
  }

  //at this point fromAr will contain sorted arr in mem
  //write this out to device in parallel
  for (i = threadId; i < n; i+=nThreads) {
    d_InArr[i] = fromArr[i];
  }
  

}


void dispFArr(float *arr, int n) {
  int i;
  for (i = 0; i < n; i++) {
    printf(" %f ", arr[i]);
  }
}


int main(int argc, char *argv[]) {

  float h_fArr[] = {0.1, 0, 0.5, 0.8, 0, 0.7, 0.8, 1.3, 0.0, 2.5, 9.10, 0, 2};
  int h_n = 13;

  //float h_fArr[] = {0.1, 0.6, 0.4, 0.3, 0.8, 2.0};
  //int h_n = 6;


  float *d_fArr;
  float *h_fSortedArr;

  int i;
  int numBits = 2;

  printf("\n");
  dispFArr(h_fArr, h_n);

  
  //allocate mem on device
  cudaMalloc((void **) &d_fArr, sizeof(float)*h_n);
  
  //copy to device
  cudaMemcpy((void *) d_fArr, (void *) h_fArr, sizeof(float)*h_n, cudaMemcpyHostToDevice);

  //sort with 2 bits at a time
  radixSort<<<1, 4, (sizeof(int)*(1<<numBits) + sizeof(float)*h_n*2)>>>(d_fArr, h_n, numBits);
  
  //copy sorted back to host
  cudaMemcpy((void *)h_fArr , (void *) d_fArr, sizeof(float)*h_n, cudaMemcpyDeviceToHost);
  
  printf("\n");
  dispFArr(h_fArr, h_n);
  printf("\n");
}
