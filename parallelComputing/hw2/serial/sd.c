/*!
\file  main.c
\brief This file is the entry point for paragon's various components
 
\date   Started 11/27/09
\author George
\version\verbatim $Id: omp_main.c 9585 2011-03-18 16:51:51Z karypis $ \endverbatim
*/


#include "simdocs.h"
#include <pthread.h>

#ifndef NTHREADS
#define NTHREADS 2
#endif



pthread_cond_t condThreadDone, condWorkReady;
pthread_mutex_t lockThreadDone, lockWorkReady;
int threadsFinished;
int workReady;

pthread_mutex_t lockCurrentlyIdle;
pthread_cond_t condCurrentlyIdle;
int currentlyIdle;

pthread_mutex_t lockCanFinish;
pthread_cond_t condCanFinish;
int canFinish;



struct ThreadData {
  int queryDoc;
  gk_csr_t *mat;
  gk_csr_t *subMat;
  gk_fkv_t *hits;
  int nhits;
  params_t *params;
};


void *findSimilarDoc(void *data) {

  int i, j, nhits;
  int32_t *marker;
  gk_fkv_t *hits, *cand;
  FILE *fpout;
  int numRowsToWork;
  params_t *params;
    
  //index or row number of matrix to work on
  int queryDoc, prevQueryDoc;
  gk_csr_t *mat;
  gk_csr_t *subMat;
  
  //get the data passed to the thread
  struct ThreadData *threadData = (struct ThreadData *) data;
  mat = threadData->mat;
  subMat = threadData->subMat;
  params = threadData->params;

  /* allocate memory for the necessary working arrays */
  hits   = gk_fkvmalloc(subMat->nrows, "ComputeNeighbors: hits");
  marker = gk_i32smalloc(subMat->nrows, -1, "ComputeNeighbors: marker");
  cand   = gk_fkvmalloc(subMat->nrows, "ComputeNeighbors: cand");

  threadData->hits = hits;
  
  prevQueryDoc = -1;
  queryDoc = threadData->queryDoc; 
  while (1) {

    //set thread as idle and signal to main thread, 
    //all threads should be idle before starting
    pthread_mutex_lock(&lockCurrentlyIdle);
    currentlyIdle++;
    pthread_cond_signal(&condCurrentlyIdle);
    pthread_mutex_unlock(&lockCurrentlyIdle);
	
    //wait for restart signal from main thread for execution
    pthread_mutex_lock(&lockWorkReady);
    while (!workReady) {
      //TODO: is prev query equal to curr query check necessary
      pthread_cond_wait(&condWorkReady, &lockWorkReady);
    }
    pthread_mutex_unlock(&lockWorkReady);

    //get the new query doc
    queryDoc = threadData->queryDoc; 
    
    /* find the best neighbors for the query document */
    if (params->verbosity > 0)
      printf("Working on query %7d\n", queryDoc);
    
    /* find the neighbors of the ith document */ 
    nhits = gk_csr_GetSimilarRows(subMat, 
				  mat->rowptr[queryDoc+1]-mat->rowptr[queryDoc], 
				  mat->rowind+mat->rowptr[queryDoc], 
				  mat->rowval+mat->rowptr[queryDoc], 
				  GK_CSR_COS, params->nnbrs, params->minsim, hits, 
				  marker, cand);

    threadData->nhits = nhits;
    
    //enter the mutex and increment the threads finished counter
    //get the mutex lock
    pthread_mutex_lock(&lockThreadDone);

    //increment the threads finished counter
    threadsFinished++;

    //signal the main thread
    pthread_cond_signal(&condThreadDone);
    
    //release the mutex lock
    pthread_mutex_unlock(&lockThreadDone);

    // Wait for permission to finish
    pthread_mutex_lock(&lockCanFinish);
    while (!canFinish) {
      pthread_cond_wait(&condCanFinish , &lockCanFinish);
    }
    pthread_mutex_unlock(&lockCanFinish);
    
    prevQueryDoc = queryDoc;
  }
  
  gk_free((void **)&hits, &marker, &cand, LTERM);
  
  pthread_exit(0);
}






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

  int queryDoc;
  int tempHitCount;
  int nsim;
  
  struct ThreadData threadData[NTHREADS];
  
  pthread_t p_threads[NTHREADS];
  pthread_attr_t attr;
  
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

  /* prepare threads */
  pthread_attr_init(&attr);
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

  //initialize conditional variables
  pthread_cond_init(&condThreadDone, NULL);
  pthread_cond_init(&condWorkReady, NULL);
  pthread_cond_init(&condCurrentlyIdle, NULL);
  pthread_cond_init(&condCanFinish, NULL);
  
  //initialize mutex
  pthread_mutex_init(&lockThreadDone, NULL);
  pthread_mutex_init(&lockWorkReady, NULL);
  pthread_mutex_init(&lockCurrentlyIdle, NULL);
  pthread_mutex_init(&lockCanFinish, NULL);
  
  //initialize initial conditional flags
  currentlyIdle = 0;
  workReady = 0;
  canFinish = 0;
  
  //initialize initial queryDoc
  queryDoc = 0;
  
  //divide the matrix based on nnz among threads
  //work per thread
  nnzPerThread = mat->rowptr[mat->nrows] / NTHREADS;
    
  prevRowUsed = -1;

  //create threads
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
    
    threadData[i].mat = mat;
    threadData[i].queryDoc = queryDoc;
    threadData[i].params = params;
    
    prevRowUsed = j;
    
    pthread_create(&p_threads[i], &attr, findSimilarDoc, (void *) &threadData[i]);
  }

  gk_clearwctimer(params->timer_2);
  gk_startwctimer(params->timer_2);

  gk_clearwctimer(params->timer_4);
  gk_startwctimer(params->timer_4);

  gk_clearwctimer(params->timer_3);
  gk_startwctimer(params->timer_3);
  
  //wait for threads to complete, and assign the next job
  for (i=0; i < mat->nrows; i++) {
    //wait for all threads to be idle before giving them work
    pthread_mutex_lock(&lockCurrentlyIdle);
    while (currentlyIdle != NTHREADS) {
      pthread_cond_wait(&condCurrentlyIdle, &lockCurrentlyIdle);
    }
    pthread_mutex_unlock(&lockCurrentlyIdle);

    //all threads are waiting for work, signal to start
    //prevent them from finishing
    canFinish = 0;
        
    //signal them to start working
    //signal threads to start again
    pthread_mutex_lock(&lockWorkReady);
    //assign new queries to thread/jobs
    for (j = 0; j < NTHREADS; j++) {
      threadData[j].queryDoc = i;
    }
    threadsFinished = 0;
    workReady = 1;
    pthread_cond_broadcast(&condWorkReady);
    pthread_mutex_unlock(&lockWorkReady); 
    
    //wait for all threads to complete
    pthread_mutex_lock(&lockThreadDone);
    while (threadsFinished < NTHREADS) {
      pthread_cond_wait(&condThreadDone, &lockThreadDone);
    }
    pthread_mutex_unlock(&lockThreadDone);

    //threads finished their job and are waiting for finish flag authorization
    //prevent them from starting again
    workReady = 0;
    currentlyIdle = 0;

    //process the thread outputs
    tempHitCount = 0;
    for (j = 0; j < NTHREADS; j++) {
      for (k = 0; k < threadData[j].nhits; k++) {
	hits[tempHitCount++] = threadData[j].hits[k];
      }
    }
    gk_stopwctimer(params->timer_2);


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


    /* write the results in the file */
    if (fpout) {
      //fprintf(fpout, "%8d %8d\n", i, nsim);
      for (j=0; j<nsim; j++) 
        fprintf(fpout, "%8d %8d %.3f\n", i, hits[j].val, hits[j].key);
    }
    gk_stopwctimer(params->timer_4);
    
    //allow threads to finish
    pthread_mutex_lock(&lockCanFinish);
    canFinish = 1;
    pthread_cond_broadcast(&condCanFinish);
    pthread_mutex_unlock(&lockCanFinish);
   
  }
  
  gk_stopwctimer(params->timer_1);
  

  /* cleanup and exit */
  if (fpout) gk_fclose(fpout);

  gk_csr_Free(&mat);

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
