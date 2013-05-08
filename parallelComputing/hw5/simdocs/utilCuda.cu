

void cudaCopyCSRIArr(int *h_IPtr, int *d_IPtr, int size) {
  //allocate device pointers d_Iptr
  cudaMalloc((void **) &d_IPtr, sizeof(int)*size);

  //copy h_Iptr to device
  cudaMemcpy((void *) d_IPtr, h_Iptr, sizeof(int)*size,\
	     cudaMemcpyHostToDevice);
}

void cudaCopyCSRFArr(float *h_FPtr, float *d_FPtr, int size) {
  //allocate device pointers d_Fptr
  cudaMalloc((void **) &d_FPtr, sizeof(float)*size);

  //copy h_Fptr to device
  cudaMemcpy((void *) d_FPtr, h_Fptr, sizeof(float)*size,\
	     cudaMemcpyHostToDevice);
}

/*
gk_csr_t *cuda_csr_Create()
{
  gk_csr_t *d_mat;
  
  cudaMalloc((void **) &d_mat, sizeof(gk_csr_t));

  gk_csr_Init(mat);

  return mat;
}
*/

cuda_csr_t *cudaCopyCSR(gk_csr_t *mat, int isIndexed) {
  
  //will pass this struct by value to kernel
  cuda_csr_t *h_mat;

  int *d_tempIPtr;
  float *d_tempFPtr;

  //TODO:
  int nnz;

  nnz = 0;
  
  assert(nnz != 0);

  h_mat = (cuda_csr_t *) malloc(sizeof(gk_csr_t)); 
  
  //TODO: check if this will work as it is direct assignment
  //TODO: this dont work
  cudaCopyCSRIArr(&mat->nrows, d_tempIPtr, 1);
  h_mat->nrows = d_tempIPtr;

  cudaCopyCSRIArr(&mat->ncols, d_tempIPtr, 1);
  h_mat->ncols = d_tempIPtr;

  //allocate device struct
  //d_mat = cudaMalloc((void**) &d_mat, sizeof(gk_csr_t));
  
  //copy row structure

  //copy row ptr
  cudaCopyCSRIArr(mat->rowptr, d_tempIPtr, (mat->nrows)+1);
  
  //point to device pointer in host
  h_mat->rowPtr = d_tempIPtr;  

  //copy device pointers rowind
  cudaCopyCSRIArr(mat->rowind, d_tempIPtr, nnz);

  //point to device pointer in host
  h_mat->rowind = d_tempIPtr;  

  //copy device pointers rowvals
  cudaCopyCSRFArr(mat->rowval, d_tempFPtr, nnz);

  //point to device pointer in host
  h_mat->rowval = d_tempFPtr;

  if (mat->rowids) {
    //copy device pointers rowids
    cudaCopyCSRIArr(mat->rowids, d_tempIPtr, mat->nrows);

    //point to device pointer in host
    h_mat->rowids = d_tempIPtr;
  }

  if (mat->rnorms) {
    //copy device pointers rnorms
    cudaCopyCSRFArr(mat->rnorms, d_tempFPtr, mat->nrows);

    //point to device pointer in host
    h_mat->rnorms = d_tempFPtr;
  }

  if (mat->rsums) {
    //copy device pointers rsums
    cudaCopyCSRFArr(mat->rsums, d_tempFPtr, mat->nrows);

    //point to device pointer in host
    h_mat->rsums = d_tempFPtr;
  }

  if (!isIndexed) {
    return h_mat;
  }
  //*** do same for column indexed
  //TODO: check whether column indexing actually necessary then only copy

  assert(mat->colptr != NULL);

  //copy device pointers colptr
  cudaCopyCSRIArr(mat->colptr, d_tempIPtr, (mat->ncols)+1);

  //point to device pointer in host
  h_mat->colPtr = d_tempIPtr;  

  //allocate device pointers colind
  cudaCopyCSRIArr(mat->colind, d_tempIPtr, nnz);

  //point to device pointer in host
  h_mat->colind = d_tempIPtr;  

  //copy device pointers colvals
  cudaCopyCSRFArr(mat->colval, d_tempFPtr, nnz);

  //point to device pointer in host
  h_mat->colval = d_tempFPtr;

  if (mat->colids) {
    //copy device pointers colids
    cudaCopyCSRIArr(mat->colids, d_tempIPtr, mat->ncols);

    //point to device pointer in host
    h_mat->colids = d_tempIPtr;
  }

  if (mat->cnorms) {
    //copy device pointers rnorms
    cudaCopyCSRFArr(mat->cnorms, d_tempFPtr, mat->ncols);

    //point to device pointer in host
    h_mat->cnorms = d_tempFPtr;
  }

  if (mat->csums) {
    //copy device pointers rsums
    cudaCopyCSRFArr(mat->csums, d_tempFPtr, mat->ncols);

    //point to device pointer in host
    h_mat->csums = d_tempFPtr;
  }

  return h_mat;
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

  /* copy the row structure */
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
