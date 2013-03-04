/*!
\file  main.c
\brief This file is the entry point for paragon's various components
 
\date   Started 11/27/09
\author George
\version\verbatim $Id: omp_main.c 9585 2011-03-18 16:51:51Z karypis $ \endverbatim
*/


#include "simdocs.h"
#include <omp.h>

#ifndef NTHREADS
#define NTHREADS 8
#endif


struct ThreadData {
  gk_csr_t *subMat;
  gk_fkv_t *hits, *cand;
  int32_t *marker;
  int nhits;
};


/*************************************************************************/
/*! Reads and computes the neighbors of each document */
/**************************************************************************/
void ComputeNeighbors(params_t *params)
{
  int i, j, k;
  gk_csr_t *mat;
  gk_fkv_t *hits;
  FILE *fpout;

  int nnzCount;
  int nnzPerThread;
  int remainingNNZ;
  int prevRowUsed; 
  int prevNNZCount;
  int tempNNZCount;

  int tempHitCount;
  int nsim;
  int numLastChunkRows;
  
  struct ThreadData threadData[NTHREADS];
  
  printf("Reading data for %s...\n", params->infstem);

  mat = gk_csr_Read(params->infstem, GK_CSR_FMT_CSR, 1, 0);

  printf("#docs: %d, #nnz: %d.\n", mat->nrows, mat->rowptr[mat->nrows]);
  
  //get the nonzero count
  nnzCount = mat->rowptr[mat->nrows];
    
  /* compact the column-space of the matrices */
  gk_csr_CompactColumns(mat);

  /* perform auxiliary normalizations/pre-computations based on similarity */
  gk_csr_Normalize(mat, GK_CSR_ROW, 2);

  /* allocate memory for the necessary working arrays */
  hits   = gk_fkvmalloc(params->nnbrs*NTHREADS, "ComputeNeighbors: hits");
  
  /* create the inverted index */
  gk_csr_CreateIndex(mat, GK_CSR_COL);

  
  /* create the output file */
  fpout = (params->outfile ? gk_fopen(params->outfile, "w",\
				      "ComputeNeighbors: fpout") : NULL);

  //divide the matrix based on nnz among threads
  //work per thread
  nnzPerThread = mat->rowptr[mat->nrows] / NTHREADS;
    
  prevRowUsed = -1;

  //create data for each thread
  gk_startwctimer(params->timer_1);
  for (i=0; i < NTHREADS; i++) {

    if (prevRowUsed == -1) {
      prevNNZCount = 0;
    } else {
      //prev row used represent the last row used in iteration
      prevNNZCount = mat->rowptr[prevRowUsed + 1];
    }
    
    //assign the suitable number of rows to the  thread
    for (j = prevRowUsed + 1; j < mat->nrows; j++) {
      tempNNZCount = mat->rowptr[j] - prevNNZCount;
      if (tempNNZCount >= nnzPerThread) {
	break;
      }
    }

    //decrement to get the right j that can fit
    j--;

    //prepare data for thread i
    //assign the submatrix to work on
    if (i == NTHREADS - 1) {
      //last thread then assign all remaining rows
      threadData[i].subMat = gk_csr_ExtractSubmatrix(mat, prevRowUsed + 1, \
					      (mat->nrows - 1) - (prevRowUsed));
    } else {
      //extract the submatrix starting from prevRowUsed + 1 row till j-1 row	
      //and [(j - 1) - prevRowUsed + 1)] rows 
      threadData[i].subMat = gk_csr_ExtractSubmatrix(mat, prevRowUsed + 1, \
						     j - (prevRowUsed));
    }
    
    //create the inverted index
    gk_csr_CreateIndex(threadData[i].subMat, GK_CSR_COL);

    //create the hits, marker and cand for the thread
    /* allocate memory for the necessary working arrays */
    threadData[i].hits   = gk_fkvmalloc(threadData[i].subMat->nrows, "ComputeNeighbors: hits");
    threadData[i].marker = gk_i32smalloc(threadData[i].subMat->nrows, -1, "ComputeNeighbors: marker");
    threadData[i].cand   = gk_fkvmalloc(threadData[i].subMat->nrows, "ComputeNeighbors: cand");
    
    prevRowUsed = j;
  }

  
  //get similarity of each doc 
  for (i=0; i < mat->nrows; i++) {
    gk_startwctimer(params->timer_2);

#pragma omp parallel for default(none) private(j) shared(threadData, mat, i, params) num_threads(NTHREADS)
    for (j = 0; j < NTHREADS; j++) {
      /* find the neighbors of the ith document */ 
      threadData[j].nhits = gk_csr_GetSimilarRows(threadData[j].subMat, 
						  mat->rowptr[i+1]-mat->rowptr[i], 
						  mat->rowind+mat->rowptr[i], 
						  mat->rowval+mat->rowptr[i], 
						  GK_CSR_COS, params->nnbrs,
						  params->minsim, threadData[j].hits, 
						  threadData[j].marker, threadData[j].cand);
      
    }


    gk_stopwctimer(params->timer_2);


    gk_startwctimer(params->timer_3);
    //process the thread outputs
    tempHitCount = 0;
    numLastChunkRows = 0;
    for (j = 0; j < NTHREADS; j++) {
      for (k = 0; k < threadData[j].nhits; k++) {
	hits[tempHitCount] = threadData[j].hits[k];
	hits[tempHitCount].val += numLastChunkRows;  
	tempHitCount++;
      }
      numLastChunkRows += threadData[j].subMat->nrows;
    }


    nsim = 0;
    //sort the hits
    if (params->nnbrs == -1 || params->nnbrs >= tempHitCount) {
      //all similarities required or 
      //similarity required > num of candidates remain after prunning
      nsim = tempHitCount;
    } else {
      nsim = gk_min(tempHitCount, params->nnbrs);
      gk_dfkvkselect(tempHitCount, nsim, hits);
      gk_fkvsortd(nsim, hits);
    }
    gk_stopwctimer(params->timer_3);


    gk_startwctimer(params->timer_4);
    /* write the results in the file */
    if (fpout) {
      //fprintf(fpout, "%8d %8d\n", i, nsim);
      for (j=0; j < nsim; j++) 
        fprintf(fpout, "%8d %8d %.3f\n", i, hits[j].val, hits[j].key);
    }
    gk_stopwctimer(params->timer_4);
    
  }
  
  gk_stopwctimer(params->timer_1);
  

  /* cleanup and exit */
  if (fpout) gk_fclose(fpout);

  gk_csr_Free(&mat);
  
  //TODO: clean up of submatrix
  
  return;
}


/*************************************************************************/
/*! This is the entry point for finding simlar patents */
/**************************************************************************/
int main(int argc, char *argv[])
{
  params_t params;
  int rc = EXIT_SUCCESS;

  cmdline_parse(&params, argc, argv);

  printf("********************************************************************************\n");
  printf("sd (%d.%d.%d) Copyright 2011, GK.\n", VER_MAJOR, VER_MINOR, VER_SUBMINOR);
  printf("  nnbrs=%d, minsim=%.2f\n",
      params.nnbrs, params.minsim);

  gk_clearwctimer(params.timer_global);
  gk_clearwctimer(params.timer_1);
  gk_clearwctimer(params.timer_2);
  gk_clearwctimer(params.timer_3);
  gk_clearwctimer(params.timer_4);

  gk_startwctimer(params.timer_global);

  ComputeNeighbors(&params);

  gk_stopwctimer(params.timer_global);

  printf("    wclock: %.2lfs\n", gk_getwctimer(params.timer_global));
  printf("    timer1: %.2lfs\n", gk_getwctimer(params.timer_1));
  printf("    timer2: %.2lfs\n", gk_getwctimer(params.timer_2));
  printf("    timer3: %.2lfs\n", gk_getwctimer(params.timer_3));
  printf("    timer4: %.2lfs\n", gk_getwctimer(params.timer_4));
  printf("********************************************************************************\n");

  exit(rc);
}
