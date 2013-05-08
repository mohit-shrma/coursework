#include <stdio.h>

#define THREADS_PER_BLOCK 128
#define NUM_BLOCKS 1


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

/*
gk_csr_t *cuda_csr_ExtractSubmatrix(gk_cst_t *mat, int rstart, int nrows) {
  
  int i;
  gk_csr_t *d_nmat;

  if (rstart+nrows > mat->nrows)
    return NULL;

  d_nmat = cuda_csr_Create();

  d_nmat->nrows  = nrows;
  d_nmat->ncols  = mat->ncols;

  //copy the row structure 
  if (mat->rowptr) {
    cudaMalloc((void **) &d_nmat->rowptr, sizeof(int) * (nrows + 1));
    cudaMemcpy((void *) d_nmat->rowptr, mat->rowptr+rstart, nrows+1);
  }

  //TODO: can we access device mem here at host, won't this be inefficient
  for (i=nrows; i>=0; i--)
    d_nmat->rowptr[i] -= d_nmat->rowptr[0];

  ASSERT(d_nmat->rowptr[0] == 0);

  if (mat->rowids) {
    cudaMalloc((void **) &d_nmat->rowids, sizeof(int)*nrows);
    cudaMemcpy((void *) d_nmat->rowids, mat->rowids+rstart, nrows);
  }

  if (mat->rnorms) {
    cudaMalloc((void **) &d_nmat->rnorms, sizeof(float)*nrows);
    cudaMemcpy((void *) d_nmat->rnorms, mat->rnorms+rstart, nrows);
  }

  if (mat->rsums) {
    cudaMalloc((void **) &d_nmat->rsums, sizeof(float)*nrows);
    cudaMemcpy((void *) d_nmat->rsums, mat->rsums+rstart, nrows);
  }

  //TODO: again accessing device mem here
  ASSERT(d_nmat->rowptr[nrows] == mat->rowptr[rstart+nrows]-mat->rowptr[rstart]);

  if (mat->rowind) {
    cudaMalloc((void **) &d_nmat->rowind, \
	       sizeof(float)*mat->rowptr[rstart+nrows]-mat->rowptr[rstart]);
    cudaMemcpy((void *) d_nmat->rowind, mat->rowind+mat->rowptr[rstart],\
	       mat->rowptr[rstart+nrows]-mat->rowptr[rstart]);
  }

  if (mat->rowval) {
    cudaMalloc((void **) &d_nmat->rowval,\
	       sizeof(float)*mat->rowptr[rstart+nrows]-mat->rowptr[rstart]);
    cudaMemcpy((void *) d_nmat->rowval, mat->rowval+mat->rowptr[rstart],\
	       mat->rowptr[rstart+nrows]-mat->rowptr[rstart]);
  }

  return d_nmat;

}
*/

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



__global__ void cudaFindNeighbors(cuda_csr_t d_QMat,
				  cuda_csr_t d_DMat,
				  float *d_sim) {
  
  int ii, i, j, k;
  
  //query doc INDEX
  int blockId = blockIdx.x; 

  //threadId with in a block, DMat doc to start with
  int thId = threadIdx.x; 
  //number of threads in blocks
  int nThreads = blockDim.x;

  int nQTerms, nSim;
  int *qInd, *colptr, *colind;
  float *qVal, *colval;

  //cuda_csr_t *dQMat = &d_QMat;
  //cuda_csr_t *dDMat = &d_DMat;

  extern __shared__ float sim[];
  //TODO: allocate this on invocation
  //also how to allocate float of size DMat->nrows

  //__shared__ float sim[]; //nrows
  //__shared__ int simPred[]; //nrows

  //float *sim = &s[0]; // shred mem for floats

  //TODO:key is similarity computed and value is doc with which similarity
  //__shared__ gk_fkv_t cand[];
  
  int countKeyVal;


  /*if (thId == 0) {
    printf("\nQ num rows: %d", *(d_QMat.nrows));
    printf("\nD num rows: %d", *(d_DMat.nrows));
    }*/

  
  for (i = 0; i < *(d_DMat.nrows); i+=nThreads) {
    sim[i] = 0.0;
  }


  //TODO: verify small p
  colptr = d_DMat.colptr;
  colind = d_DMat.colind;
  colval = d_DMat.colval;

  //get number of terms in doc blockId
  nQTerms = d_QMat.rowptr[blockId+1] - d_QMat.rowptr[blockId];
  //get row indices of doc blockId
  qInd = d_QMat.rowind + d_QMat.rowptr[blockId];
  //get nz values of doc blockId
  qVal = d_QMat.rowval + d_QMat.rowptr[blockId];
  
  //marker to mark the candidates non-zero val
  
  //store non-zero hits and partial sum

  
  //for each query nnz do multiplications in parallel
  for (i = 0; i < nQTerms; i++) {
    //get non-zero col index of term in row
    j = qInd[i];
    //perform multiplication with all elements of column j
    for (k = colptr[j]+thId; k < colptr[j+1]; k+=nThreads) {
      //similarity doc colind[k] 
      sim[colind[k]] += colval[k] * qVal[ii];
    }
  }

  //copy the similarities to device
  for (i = 0; i < *(d_DMat.nrows); i++) {
    d_sim[i] = sim[i];
  }
  

  /*
  //compact the learned sim arrays and put it in key-val struct
  //find the non-zero indices here by  scan
  preScan(sim, simPred, nrows);
  
  //scatter non-zero into simPred indices
  for (i = thId, j = 0; i < *(d_DMat.nrows); i+=nThreads) {
    if (sim[i] != 0.0f) {
      //write key-val at location simPred[i]
      cand[simPred[i]].key = sim[i];
      cand[simPred[i]].val = i;
    }
  }
  
  //total non-zero key val -> simPred[nrows]+1
  if (sim[nrows-1] != 0.0f) {
    countKeyVal = simPred[nrows-1] + 1;
  } else  {
    countKeyVal = simPred[nrows-1];
  }
  */
  //perform radix sort by keys


}




/*************************************************************************/
/*! Top-level routine for computing the neighbors of each document */
/**************************************************************************/
void cudaComputeNeighbors(params_t *params)
{
  int i, j, qID, dID, nqrows, ndrows;
  vault_t *vault;
  gk_csr_t *mat;
  sim_t **allhits;
  int *nallhits;
  FILE *fpout;

  cuda_csr_t d_QMat; //query chunk
  cuda_csr_t d_DMat; //reference/ compared against query chunk
  float *d_sim;//to comtain computed similarity values on device
  float *h_sim;//to contain computed similarity calues locally
  gk_csr_t *h_QMat;
  gk_csr_t *h_DMat;

  printf("Reading data for %s...\n", params->infstem);

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

  //allocate global memory for computed similarity chunk 
  cudaMalloc((void **) &d_sim, sizeof(float)*params->ndrows);
  h_sim = (float *)malloc(sizeof(float)*params->ndrows);
  memset(h_sim, 0, sizeof(float)*params->ndrows);

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

      //pass these matrices by value
      //dim3 dimBlock(NUM_THREADS_PER_BLOCK);//32 or avg number of nnz in columns
      //dim3 dimGrid(NUM_BLOCKS); // number of query row in chunk
      //kernel<<<dimGrid, dimBlock>>>(d_QMat, d_DMat);

     
      cudaFindNeighbors<<<NUM_BLOCKS, THREADS_PER_BLOCK, params->ndrows*sizeof(float)>>>(d_QMat, d_DMat, d_sim);
      
      //copy back to local mem
      cudaMemcpy((void *)d_sim, (void*)h_sim, params->ndrows*sizeof(float),
		 cudaMemcpyDeviceToHost);

      //write the results to file
      if (fpout) {
	for (i = 0; i < 20; i++) {
	  fprintf(fpout, "%8d %8d %.3f\n", qID, dID+i, h_sim[i]);
	}
      }
	
      gk_stopwctimer(params->timer_3);

      gk_csr_Free(&vault->pmat);
    }

    //cuda copy knn for the query

    /* write the results in the file */
    /*if (fpout) {
      for (i=0; i<nqrows; i++) {
        for (j=0; j<nallhits[i]; j++) {
          fprintf(fpout, "%8d %8d %.3f\n", qID+i, allhits[i][j].pid, allhits[i][j].sim.f);
        }
      }
    }*/

  }

  gk_stopwctimer(params->timer_1);

  /* cleanup and exit */
  if (fpout) gk_fclose(fpout);
  
  gk_csr_Free(&h_DMat);
  gk_csr_Free(&h_QMat);

  FreeVault(vault);

  free(d_sim);
  
  //free cuda mem
  cudaFree(d_sim);

  return;
}



/*
__device__ void preScan(float *sim, int *simPred, int n) {
  int ai, bi;
  int thId = threadIdx.x;
  int d = 0, offset = 1;
  int temp;

  if (sim[2*thId] != 0.0f) {
    simPred[2*thId] = 1;
  } else {
    simPred[2*thId] = 0;
  }

  if (sim[2*thId + 1] != 0.0f) {
    simPred[2*thId + 1] = 1;
  } else {
     simPred[2*thId + 1] = 0;
  }

  //build sum in place
  for (d = n>>1; d > 0; d >>=1) {
    __syncthreads();
    if (thId < d) {
      ai = offset*(2*thId+1) - 1;
      bi = offset*(2*thId+2) - 1;
      simPred[bi] += simPred[ai];
    }
    offset*=2;
  }
  
  //clear last element
  if (thId == 0) {
    simPred[n-1] = 0;
  }

  //traverse down tree & build scan
  for (int d = 1; d < n; d *=2) {
    offset >> = 1;
    __syncthreads();
    if (thId < d) {
      ai = offset*(2*thId + 1) - 1;
      bi = offset*(2*thId + 2) - 1;
      temp = simPred[ai];
      simPred[ai] = simPred[bi];
      simPred[bi] += temp;
    }
  }

  __syncthreads();
  
}




__device__ int cuda_csr_GetSimilarRows() {
  
}

*/


 
 
