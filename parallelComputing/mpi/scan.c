/*
 * this program will implement mpi scan 
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval tv;
struct timeval tz;

double getTime() {
  double currTime;
  gettimeofday(&tv, &tz);
  currTime = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
  return currTime;
}


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
  //fill array with random number between 0 to 1000
  for (i = 0; i < numCount; i++) {
    *(*nums + i) = rand_r(&myRank) % 1001;
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
  fflush(stdout);
}




//implements custom mpi scan
int myMPI_Scan(int *send, int *recv, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

  unsigned int numProcs, myRank;
  int tag, i;
  MPI_Status status;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  tag = 100;


  //printf("From process %d out of %d, Hello custom scan!\n", myRank, numProcs);

  if (myRank == 0) {
    
    if (numProcs > 1) {
      //send to next rank
      //printf("\n %d sending to next %d ..", myRank, myRank+1);
      MPI_Send(send, count, datatype, myRank + 1, tag, comm);
    }   
 
    //recv will be same as send
    for (i = 0; i < count; i++) {
      recv[i] = send[i];
    }

  } else {
    //recv from the previous rank
    //printf("\n %d receiving from prev %d ..", myRank, myRank-1);
    MPI_Recv(recv, count, datatype, myRank - 1, tag, comm, &status);
    //add send to received 
    for (i = 0; i < count; i++) {
      recv[i] += send[i];
    }
    if (myRank < numProcs - 1) {
      //send the new rev to next rank
      //printf("\n %d sending to next %d ..", myRank, myRank+1);
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

  double startTime, endTime;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  //printf("From process %d out of %d, Hello World!\n", myRank, numProcs);

  if (argc == 2) {
    //printf("%s\n", argv[0]);
    //printf("\n i/p file not passed, will randomly generate inputs\n");
    numLines = atoi(argv[1]);
    fillWithRandom(&nums, numLines);
  } else if (argc == 3){
    fileName = argv[2];
    numLines = readNums(fileName, &nums);
  } else {
    printf("%s\n", argv[0]);
    printf("\nInsufficient args\n");
    return 1;
  }

    
   //initialize results buffer
  res = (int *) malloc(sizeof(int) * numLines);
  resDup = (int *) malloc(sizeof(int) * numLines);
 
 
  if (myRank == 0) {
    printf("\n displaying input arrays:");
  }

  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  displayArr(nums, numLines);


  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  startTime = getTime();
  
  //perform inbuilt mpi scan
  MPI_Scan(nums, res, numLines, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  
  endTime = getTime();

  if (myRank == 0) {
    printf("\nTime taken for mpi scan: %1f", endTime - startTime);
    printf("\nDisplaying results of scan :");
  }
  fflush(stdout);
  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  
  displayArr(res, numLines);

  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
   
  startTime = getTime();
  //perform the custom scan
  myMPI_Scan(nums, resDup, numLines, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  
  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  endTime = getTime();

  if (myRank == 0) {
    printf("\nTime taken for custom scan: %1f", endTime - startTime);
    printf("\nDisplaying results of custom scan :");
  }
  fflush(stdout);
  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  displayArr(resDup, numLines);

  MPI_Finalize();

  //free allocate memory
  free(nums);
  free(res);
  free(resDup);

  return 0;
  
}
