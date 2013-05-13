#include <stdio.h>

#define THREADS_PER_BLOCK 4
#define NUM_BLOCKS 1
#define NUM_BITS 2
#define DEBUG 0
extern "C" 
{
#include "simdocs.h"
}


int* cudaCopyCSRIArr(int *h_IPtr,  int size) {
  int *d_IPtr;

  //allocate device pointers d_Iptr
  cudaMalloc((void **) &d_IPtr, sizeof(int)*size);

  //copy h_Iptr to device
  cudaMemcpy((void *) d_IPtr, h_IPtr, sizeof(int)*size,	\
	     cudaMemcpyHostToDevice);
  return d_IPtr;
}


float* cudaCopyCSRFArr(float *h_FPtr,  int size) {
  float *d_FPtr;

  //allocate device pointers d_Fptr
  cudaMalloc((void **) &d_FPtr, sizeof(float)*size);

  //copy h_Fptr to device
  cudaMemcpy((void *) d_FPtr, h_FPtr, sizeof(float)*size,	\
	     cudaMemcpyHostToDevice);
  return d_FPtr;
}


cuda_csr_t cudaCopyCSR(gk_csr_t *mat, int isIndexed) {
  
  cuda_csr_t d_mat;

  //will pass this struct by value to kernel
  //cuda_csr_t *h_mat = &hMat;

  int *d_tempIPtr;
  float *d_tempFPtr;

  //TODO:
  int nnz;

  nnz = mat->rowptr[mat->nrows] - mat->rowptr[0];
  
  assert(nnz != 0);

  //h_mat = (cuda_csr_t *) malloc(sizeof(gk_csr_t)); 
  
  //TODO: check if this will work as it is direct assignment
  //TODO: this dont work
  d_tempIPtr = cudaCopyCSRIArr(&(mat->nrows), 1);
  d_mat.nrows = d_tempIPtr;

  d_tempIPtr = cudaCopyCSRIArr(&(mat->ncols), 1);
  d_mat.ncols = d_tempIPtr;

  //allocate device struct
  //d_mat = cudaMalloc((void**) &d_mat, sizeof(gk_csr_t));
  
  //copy row structure

  //copy row ptr
  d_tempIPtr = cudaCopyCSRIArr(mat->rowptr,  (mat->nrows)+1);
  
  //point to device pointer in host
  d_mat.rowptr = d_tempIPtr;  

  //copy device pointers rowind
  d_tempIPtr = cudaCopyCSRIArr(mat->rowind,  nnz);

  //point to device pointer in host
  d_mat.rowind = d_tempIPtr;  

  //copy device pointers rowvals
  d_tempFPtr =  cudaCopyCSRFArr(mat->rowval,  nnz);

  //point to device pointer in host
  d_mat.rowval = d_tempFPtr;

  if (mat->rowids) {
    //copy device pointers rowids
    d_tempIPtr = cudaCopyCSRIArr(mat->rowids,  mat->nrows);

    //point to device pointer in host
    d_mat.rowids = d_tempIPtr;
  }

  if (mat->rnorms) {
    //copy device pointers rnorms
    d_tempFPtr = cudaCopyCSRFArr(mat->rnorms,  mat->nrows);

    //point to device pointer in host
    d_mat.rnorms = d_tempFPtr;
  }

  if (mat->rsums) {
    //copy device pointers rsums
    d_tempFPtr = cudaCopyCSRFArr(mat->rsums, mat->nrows);

    //point to device pointer in host
    d_mat.rsums = d_tempFPtr;
  }

  if (!isIndexed) {
    return d_mat;
  }
  //*** do same for column indexed
  //TODO: check whether column indexing actually necessary then only copy

  assert(mat->colptr != NULL);

  //copy device pointers colptr
  d_tempIPtr = cudaCopyCSRIArr(mat->colptr, (mat->ncols)+1);

  //point to device pointer in host
  d_mat.colptr = d_tempIPtr;  

  //allocate device pointers colind
  d_tempIPtr = cudaCopyCSRIArr(mat->colind, nnz);

  //point to device pointer in host
  d_mat.colind = d_tempIPtr;  

  //copy device pointers colvals
  d_tempFPtr = cudaCopyCSRFArr(mat->colval, nnz);

  //point to device pointer in host
  d_mat.colval = d_tempFPtr;

  if (mat->colids) {
    //copy device pointers colids
    d_tempIPtr = cudaCopyCSRIArr(mat->colids,  mat->ncols);

    //point to device pointer in host
    d_mat.colids = d_tempIPtr;
  }

  if (mat->cnorms) {
    //copy device pointers rnorms
    d_tempFPtr = cudaCopyCSRFArr(mat->cnorms, mat->ncols);

    //point to device pointer in host
    d_mat.cnorms = d_tempFPtr;
  }

  if (mat->csums) {
    //copy device pointers rsums
    d_tempFPtr = cudaCopyCSRFArr(mat->csums, mat->ncols);

    //point to device pointer in host
    d_mat.csums = d_tempFPtr;
  }
  
  return d_mat;
}


/* free cuda resources */
void freeCudaCSR(cuda_csr_t *mat) {
  cudaFree(mat->nrows);
  cudaFree(mat->ncols);
  cudaFree(mat->rowptr);  cudaFree(mat->colptr);  cudaFree(mat->rowids);
  cudaFree(mat->rowind);  cudaFree(mat->colind);  cudaFree(mat->colids);
  cudaFree(mat->rowval);  cudaFree(mat->colval);
  cudaFree(mat->rnorms);  cudaFree(mat->cnorms);
  cudaFree(mat->rsums);   cudaFree(mat->csums);
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


//assuming no. of threads is power of 2
//for best performance simPred is also power of 2
__device__ void compact(float *sim, int *simPred, int n, float minSim) {
  
  int i, temp, j, prev;

  //threadId with in a block, DMat doc to start with
  int thId = threadIdx.x; 

  //number of threads in blocks
  int nThreads = blockDim.x;

  __shared__ int compactCount;

  compactCount = 0;
  
  __syncthreads();  
  for (i = thId; i < n; i+= nThreads) {
    if (sim[i] >= minSim ) {
      atomicAdd(&compactCount, 1);
      simPred[i] = 1;
    }
  }
  
  if (thId == 0)
    printf("\nbefore compaction compCount:%d: ", compactCount);
  for (i = thId; i < n; i+= nThreads) {
    if (sim[i] >= minSim ) {
      printf("simPred[%d]=%d ", i, simPred[i]);
    }
  }  


  //divide the simpred into blocks,
  //scan each block in parallel, with next iteration using results from prev blocks
  scan(simPred, n);

  if (thId == 0)
    printf("\nafter compaction");
  for (i = thId; i < n; i+= nThreads) {
    if (sim[i] >= minSim ) {
      printf("simPred[%d]=%d ", i, simPred[i]);
    }
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
    //reset histogram
    zeroedInt(aggHisto, bucketSize);

    if (threadId == 0 && DEBUG) {
      printf("\n fromKeysay b4 histo :  ");
      d_dispFArr(fromKeys, n);
    }

    //aggregate in histogram in shared mem
    computeAtomicHisto(aggHisto, fromKeys, n,
		       numBits, i);

    if (threadId == 0 && DEBUG) {
      printf("\naggHisto, bitpos:%d:", i);
      dispArr(aggHisto, bucketSize);
      printf("\n fromKeysay after histo :  ");
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
      printf("\n sorted:  ");
      d_dispFArr(toKeys, n);
    }

    //toKeys contains the sorted arr, for the next iteration point fromKeys to this location
    tempFSwap = toKeys;
    toKeys = fromKeys;
    fromKeys = tempFSwap;  

    //toVals contains the sorted vals by keys,
    //for the next iteration point fromVals to this location
    tempISwap = toVals;
    toVals = fromVals;
    fromVals = tempISwap;  

  }

  //at this point fromKeys and fromVal will contain sorted arr in mem
 
}


//linearly merge two sorted keys into third one
__device__ void linearMerge(float *oldTopKeys, int *oldTopVal,
			    float *currKeys, int *currVal,
			    float *newTopKeys, int *newTopVal,
			    int k, int newArrLen) {
  int i, oldPtr, currPtr;


  for (i = 0, oldPtr=k-1, currPtr=newArrLen-1;
       i < k && oldPtr > 0 && currPtr > 0; i++) {
    if (oldTopKeys[oldPtr] > currKeys[currPtr]) {
      newTopKeys[i] = oldTopKeys[oldPtr];
      newTopVal[i] = oldTopVal[oldPtr];
      oldPtr--;
    } else {
      newTopKeys[i] = currKeys[currPtr];
      newTopVal[i] = currVal[currPtr];
      currPtr--;
    }
  }

  if (i < k) {
    //one of the array exhausted while merging
    if (currPtr < 0) {
      //exhausted current similarity space
      while (i < k && oldPtr > 0) {
	//add remaining values from old top ks
	newTopKeys[i] = oldTopKeys[oldPtr];
	newTopVal[i] = oldTopVal[oldPtr];
	oldPtr--;
	i++;
      }
    } else if (oldPtr < 0) {
      //exhausted old similarities space
      while (i < k && currPtr > 0) {
	//add remaining values from currents
	newTopKeys[i] = currKeys[currPtr];
	newTopVal[i] = currVal[currPtr];
	currPtr--;
	i++;
      }
    }
  }
  
  
}




/* d_QMat - contains query documents 
 * d_DMat - contains document to query against
 * d_sim  - previous top -k similarities, will write to this the new top-k
 * k      - number of top similarities to look for
 * 
 */
__global__ void cudaFindNeighbors(cuda_csr_t d_QMat,
				  cuda_csr_t d_DMat,
				  float *d_topK_keys, //similarity values
				  int *d_topK_vals,   //index of computed similarities
				  int k, int numBits, float minSim) {
  
  int i, j;
  
  //query doc index
  int blockId = blockIdx.x; 

  //threadId with in a block, DMat doc to start with
  int thId = threadIdx.x; 

  //number of threads in blocks
  int nThreads = blockDim.x;

  int nQTerms, nSim;
  int *qInd, *colptr, *colind;
  float *qVal, *colval;
  int countKeyVal;
  int numDMatRows, numBuckets;

  __shared__ int shNNZCount;

  extern __shared__ int s[];

  shNNZCount = 0;
  __syncthreads();

  numDMatRows  = *(d_DMat.nrows);
  numBuckets = 1 << numBits;

  //shared memory for predicates
  int *simPred = s;

  //shared memory to store similarities
  float *sim = (float*) &s[numDMatRows];
  
  //shared memory to store compacted keys & values
  float *compactKeys = (float *) &sim[numDMatRows];
  int *compactVals = (int *) &compactKeys[numDMatRows]; 
  
  //TODO
  //shared memory to store keys and values for radix sort op
  //can use previous offset as it won't be used after compaction
  float *toKeys = (float *) &compactVals[numDMatRows]; 
  int *toVals = (int*) &toKeys[numDMatRows];

  //shared memory to store histogram
  int *aggHisto = (int*) &toVals[numDMatRows];

  //shared memory storing old top keys & val
  float *oldTopKeys = (float *) &aggHisto[numBuckets]; //copy from device mem
  int *oldTopVal = (int *) &oldTopKeys[k]; //copy from device mem


  //copy to shared mem old top-k values and keys
  for (i = thId; i < k; i+= nThreads) {
    oldTopKeys[i] = d_topK_keys[i];
    oldTopVal[i] = d_topK_vals[i];
  }


  /*if (thId == 0) {
    printf("\nQ num rows: %d", *(d_QMat.nrows));
    printf("\nD num rows: %d", numDMatRows);
    }*/

  //set to zero similarity and simPreds
  for (i = thId; i < numDMatRows; i+=nThreads) {
    sim[i] = 0.0;
    simPred[i] = 0;
  }

  colptr = d_DMat.colptr;
  colind = d_DMat.colind;
  colval = d_DMat.colval;

  //get number of terms in doc blockId
  nQTerms = d_QMat.rowptr[blockId+1] - d_QMat.rowptr[blockId];
  //get row indices of doc blockId
  qInd = d_QMat.rowind + d_QMat.rowptr[blockId];
  //get nz values of doc blockId
  qVal = d_QMat.rowval + d_QMat.rowptr[blockId];
  
  //for each query nnz do multiplications in parallel
  for (i = 0; i < nQTerms; i++) {
    //get non-zero col index of term in row
    j = qInd[i];
    //perform multiplication with all elements of column j in parallel
    for (k = colptr[j]+thId; k < colptr[j+1]; k+=nThreads) {
      //similarity doc colind[k] 
      sim[colind[k]] += colval[k] * qVal[i];
    }
  }

  
  if (thId == 0) {
    printf("simPred[0] :%d, simPred[1] :%d, simPred[2] :%d",
	   simPred[0], simPred[1], simPred[2]);
  }


  //compact the learned sim arrays and put it in key-val struct
  //find the non-zero indices here by  scan
  compact(sim, simPred, numDMatRows, minSim);
  
  if (thId == 0) {
    printf("\nsimPred[0] :%d, simPred[1] :%d, simPred[2] :%d",
	   simPred[0], simPred[1], simPred[2]);
  }

  shNNZCount = 0;

  //scatter non-zero into simPred indices
  for (i = thId, j = 0; i < numDMatRows; i+=nThreads) {
    if (sim[i] >= minSim) {
      atomicAdd(&shNNZCount, 1);
      //write key-val at location simPred[i]
      compactKeys[simPred[i]] = sim[i];
      compactVals[simPred[i]] = i;
      printf("\ni=%d compactKeys[%d] = %f", i, simPred[i] , sim[i]);
    }
  }

  __syncthreads();
 
  

 

  //total non-zero key val -> simPred[nrows]+1
  if (sim[(numDMatRows)-1] != 0.0f) {
    countKeyVal = simPred[(numDMatRows)-1] + 1;
  } else  {
    countKeyVal = simPred[(numDMatRows)-1];
  }

  if (thId == 0) {
    printf("\natomic count: %d", shNNZCount);
    printf("\ncountKeyVal: %d", countKeyVal);
  }


  //before sorting
  if (thId == 0) {
    printf("\n before sorting: \n");
    for (i = 0; i < countKeyVal; i++) {
      printf(" %f %d, ", compactKeys[i], compactVals[i]);
    }
  }

  //sort compact keys and corresponding vals
  radixSort(compactKeys, toKeys,
	    compactVals, toVals,
	    aggHisto,
	    countKeyVal, numBits);

  //after sorting
  if (thId == 0) {
    printf("\n after sorting: \n");
    for (i = 0; i < countKeyVal; i++) {
      printf(" %f %d, ", compactKeys[i], compactVals[i]);
    }
  }

  //linear merge to get topk keys and vals & write it out to device
  //TODO: think of a way to do this in parallel
  linearMerge(oldTopKeys, oldTopVal,
	      compactKeys, compactVals,
	      d_topK_keys, d_topK_vals,
	      k, countKeyVal);

}




/*************************************************************************/
/*! Top-level routine for computing the neighbors of each document */
/**************************************************************************/
void cudaComputeNeighbors(params_t *params)
{
  int i, j, k, qID, dID, nqrows, ndrows;
  vault_t *vault;
  FILE *fpout;

  cuda_csr_t d_QMat; //query chunk
  cuda_csr_t d_DMat; //reference/ compared against query chunk

  //top k similarities and docs found till now
  float *h_topK_keys;
  int *h_topK_vals;

  //device space to copy top k from host
  float *d_topK_keys; 
  int *d_topK_vals;

  gk_csr_t *h_QMat;
  gk_csr_t *h_DMat;

  printf("CUDA: Reading data for %s...\n", params->infstem);

  vault = ReadData(params);

  params->endid = (params->endid == -1 ? vault->mat->nrows : params->endid);

  printf("#docs: %d, #nnz: %d.\n", vault->ndocs, vault->mat->rowptr[vault->mat->nrows]);

  /* Compact the column-space of the matrices */
  gk_csr_CompactColumns(vault->mat);

  /* Perform auxiliary normalizations/pre-computations based on similarity */
  gk_csr_Normalize(vault->mat, GK_CSR_ROW, 2);

  /* create the output file */
  fpout = (params->outfile ? gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") : NULL);

  gk_startwctimer(params->timer_1);


  //count to be selected
  k = params->nnbrs;

  //allocate device memory for computed top-k similarity chunk 
  cudaMalloc((void **) &d_topK_keys, sizeof(float)*k);
  cudaMalloc((void **) &d_topK_vals, sizeof(int)*k);

  //allocate host memory for computed top-k similarity chunk 
  h_topK_keys = (float *) malloc(sizeof(float)*k);
  h_topK_vals = (int *) malloc(sizeof(int)*k);


  /* break the computations into chunks */
  for (qID=params->startid; qID<params->endid; qID+=params->nqrows) {
    nqrows = gk_min(params->nqrows, params->endid-qID);

    if (params->verbosity > 0)
      printf("Working on query chunk: %7d, %4d\n", qID, nqrows);

    //create a copy of query chunk on cuda
    //TODO: check if need to malloc every time
    //tODO: cudaMalloc
    h_QMat = gk_csr_ExtractSubmatrix(vault->mat, qID, nqrows);

    //ASSERT(d_QMat != NULL);

    //TODO: allocate space to store top similar docs, count in cuda mem
    //reset top k for each similarity
    memset(h_topK_keys, 0, sizeof(float)*k);
    memset(h_topK_vals, 0, sizeof(int)*k);
    //reset and copy to device mem
    cudaMemcpy((void *)d_topK_keys, (void*)h_topK_keys,
	       params->ndrows*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_topK_vals, (void*)h_topK_vals,
	       params->ndrows*sizeof(int), cudaMemcpyHostToDevice);
    
    

    /* find the neighbors of the chunk */ 
    for (dID=0; dID<vault->ndocs; dID+=params->ndrows) {
      ndrows = gk_min(params->ndrows, vault->ndocs-dID);
     
      /* create the sub-matrices */
      gk_startwctimer(params->timer_2);

      //create a copy of query chunk on cuda
      //TODO: check if need to malloc every time
      //tODO: cudaMalloc
      h_DMat = gk_csr_ExtractSubmatrix(vault->mat, dID, ndrows);

      //ASSERT(d_DMat != NULL);
      gk_stopwctimer(params->timer_2);
      gk_startwctimer(params->timer_4);
      gk_csr_CreateIndex(h_DMat, GK_CSR_COL);
      gk_stopwctimer(params->timer_4);

      if (params->verbosity > 1)
        printf("  Working on db chunk: %7d, %4d, %4.2fMB\n", dID, ndrows, 
            8.0*h_DMat->rowptr[h_DMat->nrows]/(1024*1024));


      //START cuda copying and kernel invocation here
      d_QMat = cudaCopyCSR(h_QMat, 0);
      d_DMat = cudaCopyCSR(h_DMat, 1);

      /* spawn the work threads */
      gk_startwctimer(params->timer_3);

      //launch kernel
      //shared mem float:sim[ndrows],compactKeys[ndrows],toKeys[ndrows],oldTopKeys[k],
      //int:simPred[ndrows], compactVals[ndrows],oldTopVal[k],toVals[ndrows],
      //aggHisto[1<<NUM_BITS]
      cudaFindNeighbors<<<NUM_BLOCKS, THREADS_PER_BLOCK,
	(params->ndrows*3 + k)*sizeof(float) +
	(params->ndrows*3 + k + (1<<NUM_BITS) )*sizeof(int) 
	>>>(d_QMat, d_DMat, d_topK_keys, d_topK_vals,
	    k, NUM_BITS, params->minsim);
      
      //copy back to host mem
      cudaMemcpy((void *)h_topK_keys, (void*)d_topK_keys,
		 k*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy((void *)h_topK_vals, (void*)d_topK_vals,
		 k*sizeof(int), cudaMemcpyDeviceToHost);
      
      //cuda copy knn for the query
      //write the results to file
      if (fpout) {
	//TODO: zeroe the array in case prunnig to strict dont print garbage
	for (i = 0; i < k; i++) {
	  fprintf(fpout, "%8d %8d %.3f\n", qID, h_topK_vals[i], h_topK_keys[i]);
	}
      }


      gk_stopwctimer(params->timer_3);

      gk_csr_Free(&vault->pmat);
    }

    
  }

  gk_stopwctimer(params->timer_1);

  /* cleanup and exit */
  if (fpout) gk_fclose(fpout);
  
  gk_csr_Free(&h_DMat);
  gk_csr_Free(&h_QMat);

  FreeVault(vault);

  cudaFree(d_topK_keys);
  cudaFree(d_topK_vals);
  free(h_topK_keys);
  free(h_topK_vals);

  return;
}






 
 
