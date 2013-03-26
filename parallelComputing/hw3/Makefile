CC = gcc-4.7.2 
MPICC = mpicc
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point

#OpenMP options
OMPOPTIONS = -fopenmp -D__OPENMP__

#Standard Libraries
#STDLIBS = -lgcc_s -lm
STDLIBS = -lm

CFLAGS = $(COPTIONS) $(OMPOPTIONS) $(OPTFLAGS)
LIBS = $(STDLIBS)

SOURCE_MPI_SCAN = mpiscan.c
SOURCE_OMP_SCAN = ompscan.c
SOURCE_OMP_SSCAN = ompsscan.c
EXE_MPI_SCAN = mpiscan
EXE_OMP_SCAN = ompscan
EXE_OMP_SSCAN = ompsscan

default:
	$(CC) $(CFLAGS) $(LIBS) $(SOURCE_OMP_SCAN) -o $(EXE_OMP_SCAN)
	$(CC) $(CFLAGS) $(LIBS) $(SOURCE_OMP_SSCAN) -o $(EXE_OMP_SSCAN)
	$(MPICC) $(CFLAGS) $(LIBS) $(SOURCE_MPI_SCAN) -o $(EXE_MPI_SCAN)
