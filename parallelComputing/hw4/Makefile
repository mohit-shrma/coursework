CC = gcc
MPICC = mpicc
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -g -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point

#OpenMP options
OMPOPTIONS = -fopenmp -D__OPENMP__

#Standard Libraries
#STDLIBS = -lgcc_s -lm
STDLIBS = -lm

#CFLAGS = $(COPTIONS) $(OMPOPTIONS) $(OPTFLAGS)
CFLAGS = $(COPTIONS) $(OMPOPTIONS)
LIBS = $(STDLIBS)

SOURCE = sparseMatVecMPI.c matComm.c io.c
EXE = spMatVec

default:
	$(MPICC) $(CFLAGS) $(LIBS) $(SOURCE) -o $(EXE)