
CC = gcc 
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -g -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point

#Standard Libraries
STDLIBS = -fopenmp  -lm

#CFLAGS = $(COPTIONS)  $(OPTFLAGS)
CFLAGS = $(COPTIONS)
LIBS = $(STDLIBS)

SOURCE = sd.c getopt.c cmdline.c util.c
EXE = sd

default:
	$(CC) $(CFLAGS)  $(SOURCE) -o $(EXE) $(LIBS)
