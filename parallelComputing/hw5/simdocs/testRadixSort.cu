#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_BITS 2
#define DEBUG 1


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
    dispArr(arr, n);
    next = 0;

    if (i+nThreads-1 < n)
      next = arr[i+nThreads-1];
    

    if (n - i >= nThreads) {
      if (thId == 0) {
	printf("\ncalling presub scan i=%d nThreads=%d", i, nThreads);
      }
      preSubScan(arr + i, nThreads, (i>0?arr[i-1]:0) + prev);
    } else {
      //not power of 2 perform serial scan for others
      //this will be last iteration of loop

      if (thId == 0) {
	printf("\ndoing naive scan i=%d nThreads=%d", i, nThreads);
	dispArr(arr, n);
	for (j = i; j < n; j++) {
	  
	  if (j > 0)
	    temp = prev + arr[j-1];
	  else
	    temp = prev;

	  prev = arr[j];
	  arr[j] = temp;
	  printf("\ntemp=%d prev=%d arr[%d]=%d", temp, prev, j, arr[j]);
	}
	dispArr(arr, n);
      }      
    }//end else
    
    prev = next;

  }//end for

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
__device__ void writeSortedVals(int *aggHisto, float *fromKeys, float *toKeys,
				int *fromVals, int *toVals,
				int numBits, int bitpos, int n) {
  int i, key;
  int mask = (1 << numBits) - 1;
  void *vptr;
  int *iptr;

  for (i = 0; i < n; i++) {
    vptr = (void*)(fromKeys + i);
    iptr = (int*)vptr;

    key = (  (*iptr) >> bitpos) & mask;

    if (DEBUG) {
      printf("toKeys[%d] = %f\n", aggHisto[key], fromKeys[i]);
    }

    toKeys[aggHisto[key]] = fromKeys[i];
    toVals[aggHisto[key]] = fromVals[i];
    aggHisto[key]++;
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




//shared mem space for aggregated histogram
//numbits means bits at a time
__device__ void radixSort(float *fromKeys, float *toKeys,
			  int *fromVals, int *toVals,
			  int *aggHisto,
			  int n, int numBits) {
  
  int i, j, elemPerThread;

  //get current block number
  int blockId = blockIdx.x;
  
  //thread id within a block
  int threadId = threadIdx.x;

  //number of threads in block
  int nThreads = blockDim.x;

  //global thread id
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  //shared mem space to copy array to be sorted
  float *tempFSwap;
  int *tempISwap;

  //bucket size
  int bucketSize = 1 << numBits;


  if (threadId == 0 && DEBUG) {
    printf("\n fromKeys:  ");
    d_dispFArr(fromKeys, n);
  }


  //for each numbits chunk do following
  for (i = 0; i < sizeof(float)*8; i+=numBits) {

    if (threadId == 0 && DEBUG) {
      printf("\n fromKeys b4 zeroed histo :  ");
      d_dispFArr(fromKeys, n);
    }

    //reset histogram
    zeroedInt(aggHisto, bucketSize);

    if (threadId == 0 && DEBUG) {
      printf("\n fromKeys b4 histo :  ");
      d_dispFArr(fromKeys, n);
    }

    //aggregate in histogram in shared mem
    computeAtomicHisto(aggHisto, fromKeys, n,
		       numBits, i);

    if (threadId == 0 && DEBUG) {
      printf("\naggHisto, bitpos:%d:", i);
      dispArr(aggHisto, bucketSize);
      printf("\n fromKey after histo :  ");
      d_dispFArr(fromKeys, n);
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
      writeSortedVals(aggHisto, fromKeys, toKeys,
		      fromVals, toVals,
		      numBits, i, n);

    }
    __syncthreads();

    


    if (threadId == 0 && DEBUG) {
      printf("\n sorted toKeys:  ");
      d_dispFArr(toKeys, n);
    }

    //toKeys contains the sorted arr, for the next iteration point fromKeys to this location
    tempFSwap = toKeys;
    toKeys = fromKeys;
    fromKeys = tempFSwap;  

    if (threadId == 0 && DEBUG) {
      printf("\n after swap  toKeys:  ");
      d_dispFArr(toKeys, n);
      printf("\n after swap  fromKeys:  ");
      d_dispFArr(fromKeys, n);
      
    }



    //toVals contains the sorted vals by keys,
    //for the next iteration point fromVals to this location
    tempISwap = toVals;
    toVals = fromVals;
    fromVals = tempISwap;  

  }

  //at this point fromKeys and fromVal will contain sorted arr in mem
 
}





__global__ void testRadixSort(float *d_keys, int *d_vals, int n) {
  
  int i, j;

  extern __shared__ int s[];

  int numBits = NUM_BITS;

  //thread id within a block
  int thId = threadIdx.x;

  //number of threads in block
  int nThreads = blockDim.x;

  int *fromVals = s;
  float *fromKeys = (float *)&d_keys[n];
  int *toVals = (int *)&fromKeys[n];
  float *toKeys = (float *)&toVals[n];
  int *aggHisto = (int *)&toKeys[n];
  
  //copy keys and val to shared mem
  for (i = thId; i < n; i+=nThreads) {
    fromKeys[i] = d_keys[i];
    fromVals[i] = d_vals[i];
  }

  radixSort(fromKeys, toKeys, fromVals, toVals, aggHisto, n, numBits);
  
  //copy sorted values back
  for (i = thId; i < n; i+=nThreads) {
    d_keys[i] = fromKeys[i];
    d_vals[i] = fromVals[i];
  }
  
}


int main(int argc, char *argv[]) {
  
  float h_fKeys[] = {1.0, 0.4, 0.316228, 0.365148, 0.670820, 0.447214, 0.258199, 0.4,
		     0.258199, 0.316228, 0.258199, 0.258199, 0.258199, 0.258199,
		     0.258199, 0.258199};
  int h_iVal[] =    {0, 53, 54, 81, 98, 195, 283, 583, 598, 615, 654, 690, 768, 904, 919,
		     946};
  /*
  float h_fKeys[] = {1.0, 0.4, 0.316228, 0.365148, 0.670820};
  int h_iVal[] =    {0, 53, 54, 81, 98 };

  
  
  float h_fKeys[] = {1.0, 0.4, 0.3, 0.2, 0.6};
  int h_iVal[] =    {0, 53, 54, 81, 98 };
  */
  
  int n = 16;
  int i;
  int numBits = NUM_BITS;
  float *d_keys;
  int *d_val;
  
  cudaMalloc((void **) &d_keys, sizeof(float)*n);
  cudaMemcpy((void *) d_keys, (void *) h_fKeys, sizeof(float)*n, cudaMemcpyHostToDevice );
  
  cudaMalloc((void **) &d_val, sizeof(int)*n);
  cudaMemcpy((void *) d_val, (void *) h_iVal, sizeof(int)*n, cudaMemcpyHostToDevice );

  testRadixSort<<<1, 128,
    sizeof(int)*(2*n + (1<<numBits))
    + sizeof(float)*(2*n)>>>(d_keys, d_val, n);

  cudaMemcpy((void *) h_iVal, (void *) d_val, sizeof(int)*n, cudaMemcpyDeviceToHost );
  cudaMemcpy((void *) h_fKeys, (void *) d_keys, sizeof(float)*n, cudaMemcpyDeviceToHost );

  printf("\n");
  for (i = 0; i < n; i++) {
    printf(" %f %d, ", h_fKeys[i], h_iVal[i]);
  }
  printf("\n");

  return 0;
}
