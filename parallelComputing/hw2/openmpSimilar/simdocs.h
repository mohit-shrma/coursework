/*!
\file   
\brief  This file contains the various header inclusions
 
\date   Started 11/27/09
\author George
\version\verbatim $Id: simdocs.h 9500 2011-03-03 15:42:05Z karypis $ \endverbatim
*/

#ifndef _SIMDOCS_HEADER_
#define _SIMDOCS_HEADER_

/*************************************************************************
* Header file inclusion section
**************************************************************************/
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <signal.h>
#include <setjmp.h>
#include <assert.h>
#include <inttypes.h>
#include <sys/resource.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


typedef ssize_t         gk_idx_t;         /* index variable */


/*-------------------------------------------------------------
 * The following data structure stores a sparse CSR format
 *-------------------------------------------------------------*/
typedef struct {
  int nrows, ncols;
  int *rowptr, *colptr, *rowids;
  int *rowind, *colind, *colids;
  float *rowval, *colval;
  float *rnorms, *cnorms;
  float *rsums, *csums;
} gk_csr_t;


#define SCNGKIDX "zd"

/* custom signals */
#define SIGMEM                  SIGABRT
#define SIGERR                  SIGABRT

/* CSR-related defines */
#define GK_CSR_ROW      1
#define GK_CSR_COL      2

#define GK_CSR_COS      1
#define GK_CSR_JAC      2
#define GK_CSR_MIN      3

#define GK_CSR_FMT_CSR          2
#define GK_CSR_FMT_CLUTO        1

#define LTERM                   (void **) 0     /* List terminator for GKfree() */



/*-------------------------------------------------------------
 * Program Assertions
 *-------------------------------------------------------------*/
#ifndef NDEBUG
#   define ASSERT(expr)                                          \
    if (!(expr)) {                                               \
        printf("***ASSERTION failed on line %d of file %s: " #expr "\n", \
              __LINE__, __FILE__);                               \
        assert(expr);                                                \
    }

#   define ASSERTP(expr,msg)                                          \
    if (!(expr)) {                                               \
        printf("***ASSERTION failed on line %d of file %s: " #expr "\n", \
              __LINE__, __FILE__);                               \
        printf msg ; \
        printf("\n"); \
        assert(expr);                                                \
    }
#else
#   define ASSERT(expr) ;
#   define ASSERTP(expr,msg) ;
#endif 


#define GK_MKKEYVALUE_T(NAME, KEYTYPE, VALTYPE) \
typedef struct {\
  KEYTYPE key;\
  VALTYPE val;\
} NAME;\

GK_MKKEYVALUE_T(gk_fkv_t,   float,    gk_idx_t);
GK_MKKEYVALUE_T(gk_dkv_t,   double,   gk_idx_t);


/*-------------------------------------------------------------
 * CSR conversion macros
 *-------------------------------------------------------------*/
#define MAKECSR(i, n, a) \
   do { \
     for (i=1; i<n; i++) a[i] += a[i-1]; \
     for (i=n; i>0; i--) a[i] = a[i-1]; \
     a[0] = 0; \
   } while(0) 

#define SHIFTCSR(i, n, a) \
   do { \
     for (i=n; i>0; i--) a[i] = a[i-1]; \
     a[0] = 0; \
   } while(0) 



#define gk_clearwctimer(tmr) (tmr = 0.0)
#define gk_startwctimer(tmr) (tmr -= gk_WClockSeconds())
#define gk_stopwctimer(tmr)  (tmr += gk_WClockSeconds())
#define gk_getwctimer(tmr)   (tmr)


#define gk_max(a, b) ((a) >= (b) ? (a) : (b))
#define gk_min(a, b) ((a) >= (b) ? (b) : (a))

#define QSSWAP(a, b, stmp) do { stmp = (a); (a) = (b); (b) = stmp; } while (0)



void gk_csr_CompactColumns(gk_csr_t *mat);
void *gk_malloc(size_t, char *);
void *gk_realloc(void *oldptr, size_t nbytes, char *msg);
void gk_free(void **ptr1,...);
void gk_FreeMatrix(void ***r_matrix, size_t ndim1, size_t ndim2);
void gk_AllocMatrix(void ***r_matrix, size_t elmlen, size_t ndim1, size_t ndim2);
void gk_errexit(int signum, char *f_str,...);
void errexit(char *f_str,...);

int gk_dfkvkselect(size_t n, int topk, gk_fkv_t *cand);
void gk_fkvsortd(size_t n, gk_fkv_t *base);

uintmax_t gk_GetCurMemoryUsed();
uintmax_t gk_GetMaxMemoryUsed();

char *gk_strdup(char *orgstr);
FILE *gk_fopen(char *fname, char *mode, const char *msg);
void gk_fclose(FILE *fp);
int gk_fexists(char *fname);
void gk_getfilestats(char *fname, gk_idx_t *r_nlines, gk_idx_t *r_ntokens, 
        gk_idx_t *r_max_nlntokens, gk_idx_t *r_nbytes);
gk_idx_t gk_getline(char **lineptr, size_t *n, FILE *stream);
double gk_WClockSeconds(void);
uintmax_t gk_GetCurMemoryUsed();
uintmax_t gk_GetMaxMemoryUsed();
void gk_csr_Init(gk_csr_t *mat);
gk_csr_t *gk_csr_Create();
void gk_csr_CompactColumns(gk_csr_t *mat);
void gk_csr_Normalize(gk_csr_t *mat, int what, int norm);
gk_csr_t *gk_csr_Read(char *filename, int format, int readvals, int numbering);
void gk_csr_FreeContents(gk_csr_t *mat);
void gk_csr_Free(gk_csr_t **mat);
void gk_csr_CreateIndex(gk_csr_t *mat, int what);
gk_csr_t *gk_csr_ExtractSubmatrix(gk_csr_t *mat, int rstart, int nrows);
int gk_csr_GetSimilarRows(gk_csr_t *mat, int nqterms, int *qind, float *qval, 
        int simtype, int nsim, float minsim, gk_fkv_t *hits, int *i_marker,
        gk_fkv_t *i_cand);




#include "defs.h"
#include "struct.h"
#include "proto.h"

#include "gk_getopt.h"
#include "gk_mksort.h"
#include "gk_mkmemory.h"
#include "gk_mkblas.h"



GK_MKALLOC_PROTO(gk_i,   int)
GK_MKALLOC_PROTO(gk_i32, int32_t)
GK_MKALLOC_PROTO(gk_f,   float)
GK_MKALLOC_PROTO(gk_fkv,   gk_fkv_t)
GK_MKALLOC_PROTO(gk_dkv,   gk_dkv_t)


#endif
