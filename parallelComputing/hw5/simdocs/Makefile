#CC = gcc-4.7.2
CC = nvcc 
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -g -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point



#OpenMP options
OMPOPTIONS = -Xcompiler -fopenmp -D__OPENMP__ 

#Standard Libraries
#STDLIBS = -lgcc_s -lm
STDLIBS = -lm

#CFLAGS = $(COPTIONS) $(OMPOPTIONS) $(OPTFLAGS)
#CFLAGS = $(COPTIONS) $(OMPOPTIONS)
CFLAGS = -g -G $(OMPOPTIONS) 
LIBS = $(STDLIBS)

SOURCE = getopt.c all_common.c all_cmdline.c util.c omp_nbrs.c omp_main.c cuda_nbrs.cu 
EXE = cudasd

default:
	$(CC) $(CFLAGS) $(LIBS) $(SOURCE) -o $(EXE)

