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
#define NTHREADS 8
#endif


//will convert the passed number to string
char *toStr4mNum(int a) {
  char *str1, *str2;
  char *temp1, *temp2;

  str1 = (char *) malloc(10);
  str2 = (char *) malloc(10);

  temp1 = str1;

  if (a != 0) {
    while ( a != 0) {
      *temp1++ = (char)((int)('0') + (a % 10));
      a = a/10;
    }
  } else {
    *temp1++ = '0';
  }
  
  temp2 = str2;
  temp1--;
  while (temp1 != str1) {
    *temp2++ = *temp1--;
  }
  
  *temp2++ = *temp1;
  *temp2 = '\0';
  
  return str2;
}



//append two strings and return the third
char *strjoin(char *str1, char *str2) {
  char *temp1;
  char *temp2, *combined;
  int totalSize;

  //get combined size of strings
  totalSize = 0;

  temp1 = str1;
  while (*temp1 != '\0') {
    totalSize++;
    temp1++;
  }
  
  temp1 = str2;
  while (*temp1 != '\0') {
    totalSize++;
    temp1++;
  }

  //allocate memory for combined string
  combined = (char*) malloc(sizeof(char) * totalSize);

  //copy string 1
  temp1 = str1;
  temp2 = combined;
  while (*temp1 != '\0') {
    *temp2 = *temp1;
    temp2++;
    temp1++;
  }

  //copy string2
  temp1 = str2;
  while (*temp1 != '\0') {
    *temp2 = *temp1;
    temp2++;
    temp1++;
  }
  *temp2 = '\0';
  printf("\ncombined string: %s \n", combined);
  return combined;
}


struct ThreadData {
  int startDoc;
  int endDoc;
  params_t *params;
  gk_csr_t *mat;
  char *outfile;
};



void *findSimilarDoc(void *data) {

  int i, j, nhits;
  int32_t *marker;
  gk_fkv_t *hits, *cand;
  FILE *fpout;
  int numRowsToWork;
  
  //index or row number of matrix to work on
  int startDoc, endDoc;
  params_t *params;
  gk_csr_t *mat;
  
  //get the data passed to the thread
  struct ThreadData *threadData = (struct ThreadData *) data;
  startDoc = threadData->startDoc;
  endDoc = threadData->endDoc;
  params = threadData->params;
  mat = threadData->mat;

  /* create the output file */
  fpout = (threadData->outfile ? gk_fopen(threadData->outfile, "w", "ComputeNeighbors: fpout") : NULL);

  //number of rows to work
  numRowsToWork = endDoc - startDoc + 1;

  /* allocate memory for the necessary working arrays */
  hits   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: hits");
  marker = gk_i32smalloc(mat->nrows, -1, "ComputeNeighbors: marker");
  cand   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: cand");

  /* find the best neighbors for each query document */
  for (i=startDoc; i<=endDoc; i++) {
    if (params->verbosity > 0)
      printf("Working on query %7d\n", i);

    /* find the neighbors of the ith document */ 
    nhits = gk_csr_GetSimilarRows(mat, 
                 mat->rowptr[i+1]-mat->rowptr[i], 
                 mat->rowind+mat->rowptr[i], 
                 mat->rowval+mat->rowptr[i], 
                 GK_CSR_COS, params->nnbrs, params->minsim, hits, 
                 marker, cand);

    /* write the results in the file */
    if (fpout) {
      for (j=0; j<nhits; j++) 
        fprintf(fpout, "%8d %8d %.3f\n", i, hits[j].val, hits[j].key);
    }
  }


  /* cleanup and exit */
  if (fpout) gk_fclose(fpout);

  gk_free((void **)&hits, &marker, &cand, LTERM);
  
  pthread_exit(0);
}




/*************************************************************************/
/*! Reads and computes the neighbors of each document */
/**************************************************************************/
void ComputeNeighbors(params_t *params)
{
  int i;
  gk_csr_t *mat;
  FILE *fpout;

  //temporary file handle
  FILE *tempF;
  //to read stuff from file
  int ch;

  int workPerThread;
  int extraWork;
  int prevWorkEnd; 

  //used to create temporary files
  char *temp; 
  
  struct ThreadData threadData[NTHREADS];
  
  pthread_t p_threads[NTHREADS];
  pthread_attr_t attr;
  
  printf("Reading data for %s...\n", params->infstem);

  mat = gk_csr_Read(params->infstem, GK_CSR_FMT_CSR, 1, 0);

  printf("#docs: %d, #nnz: %d.\n", mat->nrows, mat->rowptr[mat->nrows]);

  /* compact the column-space of the matrices */
  gk_csr_CompactColumns(mat);

  /* perform auxiliary normalizations/pre-computations based on similarity */
  gk_csr_Normalize(mat, GK_CSR_ROW, 2);

  /* create the inverted index */
  gk_csr_CreateIndex(mat, GK_CSR_COL);

  /* create the output file */
  fpout = (params->outfile ? gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") : NULL);

  /* prepare threads */
  pthread_attr_init(&attr);
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

  //work per thread
  workPerThread = mat->nrows / NTHREADS;
  
  //extra work
  extraWork = mat->nrows % NTHREADS;
  prevWorkEnd = -1;

  //create threads
  gk_startwctimer(params->timer_1);
  for (i=0; i < NTHREADS; i++) {

    //prepare data for thread i
    threadData[i].params = params;
    threadData[i].mat = mat;
    threadData[i].startDoc = prevWorkEnd + 1;
    threadData[i].endDoc = prevWorkEnd + workPerThread;

    if (extraWork > 0) {
      threadData[i].endDoc += 1;
      extraWork--;
    }

    prevWorkEnd = threadData[i].endDoc;
    
    //create the output file for thread
    temp = toStr4mNum(i);
    threadData[i].outfile = (params->outfile ? strjoin(params->outfile, temp) : NULL); 

    pthread_create(&p_threads[i], &attr, findSimilarDoc, (void *) &threadData[i]);
  }

  //wait for threads to complete
  for (i=0; i < NTHREADS; i++) {
    pthread_join(p_threads[i], NULL);
  }
  gk_stopwctimer(params->timer_1);
  
  //combine the output files of thread and write in one file
  gk_startwctimer(params->timer_2);
  if (params->outfile) {
    /* create the output file */
    fpout = (params->outfile ? gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") : NULL);

    for (i=0; i < NTHREADS; i++) {
      //open the thread output
      temp = toStr4mNum(i);
      tempF = fopen(strjoin(params->outfile, temp), "r");
      //copy the thread output to main output
      while ((ch = getc(tempF)) != EOF) {
	putc(ch, fpout);
      }
      gk_fclose(tempF);
    }
  }
  gk_stopwctimer(params->timer_2);


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
