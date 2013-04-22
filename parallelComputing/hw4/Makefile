CC = gcc
#MPICC = mpicc
MPICC = /soft/openmpi/1.3/linux-x86_64/bin/mpicc
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99  -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point

#OpenMP options
OMPOPTIONS = -fopenmp -D__OPENMP__

#Standard Libraries
#STDLIBS = -lgcc_s -lm
STDLIBS = -lm

CFLAGS = $(COPTIONS) $(OMPOPTIONS) $(OPTFLAGS)
#CFLAGS = $(COPTIONS) $(OMPOPTIONS)
LIBS = $(STDLIBS)

SOURCE = sparseMatVecMPI.c matComm.c io.c common.c vecComm.c mult.c debug.c
EXE = matvec_mpi

default:
	$(MPICC) $(CFLAGS) $(LIBS) $(SOURCE) -o $(EXE)
