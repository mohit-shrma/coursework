#ifndef _MATT_COMM_
#define _MATT_COMM_
#include <mpi.h>
#include "common.h"
#include "io.h"
void scatterMatrix(CSRMat *csrMat, CSRMat **myCSRMat, int **rowInfo);

#endif
