/*
 * this program will implement mpi scan 
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

//read the numbers from file and return the count of nnumbers
int readNums(char *fileName, int **nums) {
  //to stor number count
  size_t numLines;
  //temp int to read num
  int x;

  int i;
  FILE *fin;

  if ((fin = fopen(fileName, "r")) == NULL) {
    fprintf(stderr, "Error: failed to open '%s' for reading\n", fileName);
    exit(1);
  } else if (fscanf(fin, "%zu\n", &numLines) != 1) {
    fprintf(stderr, "Error: failed to read first line of '%s' \n", fileName);
    fclose(fin);
    exit(2);
  }

  //allocate space numlines numbers
  *nums = (int *) malloc(sizeof(int) * numLines);

  for (i = 0; i < numLines; i++) {
    if (fscanf(fin, "%d\n", (*nums)+i) != 1) {
      fprintf(stderr, "Err: failed to read integer from line %zu/%zu in '%s'\n",\
	      i, numLines, fileName);
      fclose(fin);
      exit(3);
    }
  }
  
  fclose(fin);
  return numLines;
}

//generate num count random numbers and fill in nums
void fillWithRandom(int **nums, int numCount) {
  int i;
  unsigned int myRank;
  //get the process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  //allocate memory for arr
  *nums = (int *) malloc(sizeof(int) * numCount);
  //fill array with random number
  for (i = 0; i < numCount; i++) {
    *(*nums + i) = rand_r(&myRank);
  }
}


//routinr to display array rank by rank
void displayArr(int *arr, int arrLen) {
  
  int numProcs, myRank;
  int i, j;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  //display the results
  for ( i = 0; i < numProcs; i++) {
    if (myRank == i) {
      printf("\n %d: ", myRank);
      for (j = 0; j < arrLen; j++) {
	printf(" %d ", arr[j]);
      }
      printf("\n");
    }
  }

}




//implements custom mpi scan
int myMPI_Scan(int *send, int *recv, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

  unsigned int numProcs, myRank;
  int tag, i;
  MPI_Status status;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  tag = 100;


  printf("From process %d out of %d, Hello custom scan!\n", myRank, numProcs);

  if (myRank == 0) {
    
    if (numProcs > 1) {
      //send to next rank
      printf("\n %d sending to next %d ..", myRank, myRank+1);
      MPI_Send(send, count, datatype, myRank + 1, tag, comm);
    }   
 
    //recv will be same as send
    for (i = 0; i < count; i++) {
      recv[i] = send[i];
    }

  } else {
    //recv from the previous rank
    printf("\n %d receiving from prev %d ..", myRank, myRank-1);
    MPI_Recv(recv, count, datatype, myRank - 1, tag, comm, &status);
    //add send to received 
    for (i = 0; i < count; i++) {
      recv[i] += send[i];
    }
    if (myRank < numProcs - 1) {
      //send the new rev to next rank
      printf("\n %d sending to next %d ..", myRank, myRank+1);
      MPI_Send(recv, count, datatype, myRank + 1, tag, comm);
    }
  }

  return 1;
}



int main(int argc, char *argv[]) {
  int numProcs, myRank;
  //store the array and length of array
  int *nums, numLines;
  //input filename containing numbers
  char *fileName;

  //buff to store array for second mpi op
  int *numsDup;
  
  //buff to store results
  int *res, *resDup;
  
  int i, j;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  printf("From process %d out of %d, Hello World!\n", myRank, numProcs);

  if (argc < 2) {
    printf("%s\n", argv[0]);
    printf("\nInsufficient args\n");
    return 1;
  }
  
  fileName = argv[1];
  numLines = readNums(fileName, &nums);

  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  if (myRank == 0) {
    printf("\n displaying input arrays:");
  }
  displayArr(nums, numLines);


  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  //initialize results buffer
  res = (int *) malloc(sizeof(int) * numLines);
  resDup = (int *) malloc(sizeof(int) * numLines);

  //perform inbuilt mpi scan
  MPI_Scan(nums, res, numLines, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (myRank == 0) {
    printf("\n displaying results of scan :");
  }
  displayArr(res, numLines);

  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
   

  //perform the custom scan
  myMPI_Scan(nums, resDup, numLines, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  
  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  if (myRank == 0) {
    printf("\n displaying results of custom scan :");
  }
  displayArr(resDup, numLines);

    

  MPI_Finalize();

  //free allocate memory
  free(nums);
  free(res);
  free(resDup);

  return 0;
  
}
