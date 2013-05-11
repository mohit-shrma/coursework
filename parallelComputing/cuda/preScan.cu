#include <stdio.h>

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





//assuming no. of threads is power of 2
//for best performance simPred is also power of 2
__device__ void compact(float *sim, int *simPred, int n) {
  
  int i, temp, j, prev;

  //threadId with in a block, DMat doc to start with
  int thId = threadIdx.x; 

  //number of threads in blocks
  int nThreads = blockDim.x;

  for (i = thId; i < n; i+= nThreads) {
    if (sim[i] > 0.0f) {
      simPred[i] = 1;
    }
  }


  //divide the simpred into nThreads blocks,
  //scan each block in parallel, with next iteration using results from prev blocks
  scan(simPred, n);

}


__global__ void compaction(float *d_in, float *d_keys, int *d_val, int *d_numKeys, int n) {
  
  int i;

  //threadId with in a block, DMat doc to start with
  int thId = threadIdx.x; 

  //number of threads in blocks
  int nThreads = blockDim.x;

  extern __shared__ int s[];
  int *simPred = s;

  for (i = thId; i < n; i+=nThreads) {
    simPred[i] = 0;
  }

  compact(d_in, simPred, n);

  for (i = thId; i < n; i+= nThreads) {
    if (d_in[i] != 0.0f) {
      d_keys[simPred[i]] = d_in[i];
      d_val[simPred[i]] = i;
    }
  }
  
  if (thId == 0) {
  *d_numKeys = simPred[n-1] + (d_in[n-1] != 0.0f);
  }

}




int main(int argc, char *argv[]) {

  float h_fArr[] = {0.1, 0, 0.5, 0.8, 0, 0.7, 0.8, 1.3, 0.0, 2.5, 9.10, 0, 2};
  int h_n = 13;
  
  int *h_val;
  int *h_numKeys;
  float *h_keys;

  float *d_fArr; 
  int *d_val;
  float *d_keys;
  int *d_numKeys;


  int i;

  for (i = 0; i < h_n; i++) {
    printf(" %.2f ", h_fArr[i]);
  }


  //allocate memory on host
  h_val = (int*) malloc(sizeof(int) * h_n);
  h_keys = (float *) malloc(sizeof(float) * h_n);
  h_numKeys = (int *) malloc(sizeof(int) * 1);

  //allocate memory on device
  cudaMalloc((void **)&d_fArr, sizeof(float)*h_n);
  cudaMemcpy((void *)d_fArr, (void*)h_fArr, sizeof(float)*h_n, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_val, sizeof(int)*h_n); 
  cudaMalloc((void **)&d_keys, sizeof(float)*h_n);
  cudaMalloc((void **)&d_numKeys, sizeof(int)*1);

  //compact<<<1,4>>>(d_fArr, d_keys, h_n);
  compaction<<<1,4, sizeof(int)*h_n>>>(d_fArr, d_keys, d_val, d_numKeys, h_n);

  cudaMemcpy((void *)h_numKeys, (void*)d_numKeys, sizeof(int),
	     cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_val, (void*)d_val, sizeof(int)*(*h_numKeys),
	     cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_keys, (void*)d_keys, sizeof(float)*(*h_numKeys),
	     cudaMemcpyDeviceToHost);

 
  printf("\n");

  for (i = 0; i < (*h_numKeys); i++) {
    printf(" %d ", h_val[i]);
  }

  printf("\n");

  for (i = 0; i < (*h_numKeys); i++) {
    printf(" %.2f ", h_keys[i]);
  }


  printf("\n");

  cudaFree(d_fArr);
  cudaFree(d_keys);

  free(h_val);
  free(h_numKeys);
  free(h_keys);

  cudaFree(d_fArr);
  cudaFree(d_val);
  cudaFree(d_keys);
  cudaFree(d_numKeys);

  return 0;
}
