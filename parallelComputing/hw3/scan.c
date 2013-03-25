/*
 * this program will implement mpi scan 
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define CHUNK_SZ 1000

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
}


void localScan(int *arr, int arrLen) {
  int i;
  for (i = 1; i < arrLen; i++) {
    arr[i] = arr[i] + arr[i-1];
  }
}


int isPowerOf2(int num) {
  return num && !(num & (num-1));
}


int myMPIScan(int *arr, int count) {

  unsigned int numProcs, myRank, newRank;
  int nearPow, tempNumProcs, procsNeeded;
  int  *dupArr;
  int i, j, tag, status;
  
  MPI_Comm newComm;
  
  int color, key;

  dupArr = (int*) 0;
  tag = 100;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (isPowerOf2(numProcs)) {
    //printf("\n num of procs is power of 2");
    mySweepMPIScan(arr, count, MPI_INT, MPI_COMM_WORLD);
  } else {
    //num of processes not power of 2 
    //passed num procs not power of 2 
    //get lower nearest power of 2
    nearPow = (int)log2(numProcs);
    tempNumProcs = pow(2, nearPow);

    //printf("\n myRank:%d  tempNumProcs: %d", myRank, tempNumProcs);

    //split communicator for first tempNumProcs procs
    if (myRank < tempNumProcs) {
      color = 1;
      key = myRank;
    } else {
      color = 0;
      key = myRank;
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, key, &newComm);
    
    //perform scan on first group
    if (color == 1) {
      //perform sweep mpi scan on this group
      mySweepMPIScan(arr, count, MPI_INT, newComm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    //temp group finished sweep scan operations
    
    //need to perform scan on remaining procs
    
    //number of processors to borrow 
    procsNeeded = tempNumProcs - ( numProcs - tempNumProcs) ;
    
    //dup the result for procs that will be borrowed
    //procs with rank < procsNeeded will be borrowed
    if (myRank < procsNeeded) {
      dupArr = (int *)malloc(sizeof(int)*count);
      memcpy(dupArr, arr, sizeof(int)*count);
      //clear the array after storing result
      //TODO: not necessay as not concerned with what goes here
      memset(arr, 0, sizeof(int)*count);
    }
    

    //printf("\nnumProcs:%d tempNumProcs:%d procsNeeded:%d", numProcs, tempNumProcs, procsNeeded);
    
    if (myRank < procsNeeded || myRank >= tempNumProcs) {
      //assign group to borrowed rocs and unused procs
      color = 1;
      if (myRank < procsNeeded) {
	//increase key to put at end the already used procs
	key = myRank + numProcs;
      } else {
	key = myRank;
      }
    } else {
      color = 0;
      key = myRank;
    }

    //printf("\n sec scan myRank: %d color:%d", myRank, color);


    MPI_Comm_split(MPI_COMM_WORLD, color, key, &newComm);
  
    if (color == 1) {
      //perform sweep mpi scan on this group
      mySweepMPIScan(arr, count, MPI_INT, newComm);
      
    }
    
    
    //restore values from dupArr to arr
    if (myRank < procsNeeded) {
      memcpy(arr, dupArr, sizeof(int)*count);
      memset(dupArr, 0, sizeof(int)*count);
      //printf("\nrank:%d arr[0]:%d", myRank, arr[0]);
    }

    //make sure each procs reach this point
    MPI_Barrier(MPI_COMM_WORLD);

    //broadcast the last procs in prev scan to all others in new scan
    //form a group for broadcast
    if (myRank >= tempNumProcs - 1) {
      color = 1;
      key = myRank;
      //initialize dupArr for broadcast if null
      if (!dupArr) {
	dupArr = (int *)malloc(sizeof(int)*count);
	memset(dupArr, 0, sizeof(int)*count);
      }
    } else {
      color = 0;
      key = myRank;
    }

    //printf("\n bcast myRank: %d color:%d", myRank, color);
    
    //split the communicator into different groups
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &newComm);
    
    if (myRank == tempNumProcs -1) {
      //last proc of previous scan
      memcpy(dupArr, arr, sizeof(int)*count);
    }

    if (color == 1) {
      //broadcast arr of last proc from previous scan
      MPI_Bcast(dupArr, count, MPI_INT, 0, newComm);      
    }

    //make sure each procs reach this point
    MPI_Barrier(MPI_COMM_WORLD);

    //add value broadcasted from last proc of previous scan to arr
    if (myRank >= tempNumProcs) {
      
      for (i = 0; i < count; i++) {
	//printf("\nmyRank: %d arr[%d]=%d dupArr[%d]=%d", myRank, i, arr[i], i, dupArr[i]);
	arr[i] += dupArr[i];
      }
    }
    

  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  //printf("\nbefore free mem");
  if (dupArr != (int *)0) {
    free(dupArr);
  }
  
  return 1;
}


/*
 *Apply sweep scan across all processor on send[i]'s 
 */
int mySweepMPIScan(int *arr, int count, MPI_Datatype datatype, MPI_Comm comm) {

  unsigned int numProcs, myRank;
  int send, recv;
  int tag, temp;
  int tempPow, i, d, k, last;
  int *chunk;

  MPI_Status status;



  
  MPI_Comm_size(comm, &numProcs);
  MPI_Comm_rank(comm, &myRank);
  
  tag = 100;
  
  assert(isPowerOf2(numProcs));

  //printf("\n Number of procs: %d", numProcs);
  //printf("\n My rank: %d", myRank);
  if (isPowerOf2(numProcs)) {
    for (i = 0; i < count; i++) {
      
      if (myRank == numProcs -1) {
	//save the last proc's element as this is going to be root
	last = arr[i];
      }
      


      MPI_Barrier(comm);

      if (myRank == 0) {
	//printf("\n\nstarting up-sweep phase");
      }

      MPI_Barrier(comm);
      //apply sweep across all procs on arr[i]

      //up sweep phase
      //perform up-sweep on the arr[i]'s
      for(d = 0; d <= (int)log2(numProcs) - 1; d++) {
	//for the current level get partial sums
	tempPow = (int)pow(2, d+1);
	for (k = 0; k < numProcs; k += tempPow) {
	  //send from (k+(int)(pow(2, d))-1) to (k+tempPow-1)
	  if (myRank == (k+(int)(pow(2, d))-1)) {
	    //send to (k+tempPow-1)
	    //printf("\nsend from %d to %d", (k+(int)(pow(2, d))-1), (k+tempPow-1));
	    send = arr[i];
	    MPI_Send(&send, 1, datatype, (k+tempPow-1), tag, comm);
	  } else if (myRank == (k+tempPow-1)) {
	    //receive from (k+(int)(pow(2, d))-1)
	    //printf("\nreceive from %d to %d", (k+(int)(pow(2, d))-1), (k+tempPow-1));
	    MPI_Recv(&recv, 1, datatype, (k+(int)(pow(2, d))-1), tag, comm,\
		     &status);
	    //add to arr[i]
	    arr[i] += recv;
	  }
	}
	//make sure all procs reach this stage
	MPI_Barrier(comm);
      }
      
      //make sure all procs reach this stage
      MPI_Barrier(comm);


      if (myRank == 0) {
	//printf("\nstarting down-sweep phase");
      }


      //down sweep phase
      //perform down-sweep on arr[i]'s
      if (myRank == numProcs -1) {
	//set value for root of tree or for last proc
	arr[i] = 0;
      }

      for (d = (int)log2(numProcs) - 1; d >= 0; d--) {
	tempPow = (int)pow(2, d+1);
	//for the level propagate the reductions
	for (k = 0; k < numProcs; k += tempPow) {
	  
	  if (myRank == (k+tempPow-1)) {
	    //send from (k+tempPow-1) to (k+(int)pow(2, d)-1)
	    send = arr[i];
	    MPI_Send(&send, 1, datatype, (k+(int)pow(2, d)-1), tag, comm);

	    //receive from (k+(int)pow(2, d)-1)
	    MPI_Recv(&recv, 1, datatype, (k+(int)pow(2, d)-1), tag, comm,	\
		     &status);

	    //add recv to arr[i]
	    arr[i] += recv;
	    
	  } else if (myRank == (k+(int)pow(2, d)-1)) {
	    //receive from (k+tempPow-1)
	    MPI_Recv(&recv, 1, datatype, (k+tempPow-1), tag, comm, \
		     &status);
	    
	    //send from (k+(int)pow(2, d)-1) to (k+tempPow-1)
	    send = arr[i];
	    MPI_Send(&send, 1, datatype, (k+tempPow-1), tag, comm);

	    //replace arr[i] with value received
	    arr[i] = recv;
	  }
	}
	//make sure all procs completed this iteration
	MPI_Barrier(comm);
      }

      /*for (k = 0; k < numProcs; k++) {
	if (myRank == k) {
	  printf("\n B4 shift rank:%d val:%d", myRank, arr[i]);
	}
	}*/
      
      temp = arr[i];
      //for inclusive scan need to shift arr[i]'s
      for (k = numProcs -1; k >= 1; k--) {
	//send from procs k to k - 1
	if (myRank == k) {
	  //send from k 
	  send = temp;
	  MPI_Send(&send, 1, datatype, k-1, tag, comm);
	} else if (myRank == k-1) {
	  //recv by k-1 
	  MPI_Recv(&recv, 1, datatype, k, tag, comm, &status);
	  //replace arr[i] with recv
	  arr[i] = recv;
	}
      }

      /*for (k = 0; k < numProcs; k++) {
	if (myRank == k) {
	  printf("\n Aftr shift rank:%d val:%d", myRank, arr[i]);
	}
	}*/

      //make sure all procs at this stage
      MPI_Barrier(comm);

      //add last in orig array to last-1 elem
      if (myRank == numProcs - 1) {
	//last procs in the ranks
	recv = 0;
	if (numProcs > 1) { 
	  //recv from numProcs-2
	  MPI_Recv(&recv, 1, datatype, numProcs - 2, tag, comm,	\
		   &status);
	}
	//add recv to orig last elem
	arr[i] = recv + last;
      } else if (myRank == numProcs - 2) {
	//second last procs in ranks
	//send to numProcs - 1
	send = arr[i];
	MPI_Send(&send, 1, datatype, numProcs  - 1, tag, comm);
      }

      /*for (k = 0; k < numProcs; k++) {
	if (myRank == k) {
	  printf("\n after last handle rank:%d val:%d", myRank, arr[i]);
	}
	}*/
      
      //ANOTHER ALTERNATIVE to shift
      //gather all arr[i]'s at last proc
      //shift arr 
      //add last in orig array to last-1 elem
      //scatter the arr across all proc

    }
  } else {
  }
  
  return 1;
}
  

//implements custom mpi scan
int mySimpleMPScan(int *send, int *recv, int count, MPI_Datatype datatype,\
	       MPI_Op op, MPI_Comm comm) {

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
    //printf("\n displaying input arrays:");
  }

  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  //displayArr(nums, numLines);


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
    //printf("\nDisplaying results of scan :");
  }
  fflush(stdout);
  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);
  
  //displayArr(res, numLines);
  fflush(stdout);
  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  memcpy(resDup, nums, sizeof(int)*numLines);   
  startTime = getTime();
  //perform the custom scan
  //myMPI_Scan(nums, resDup, numLines, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  //myMPI_Collective_Scan(nums, resDup, numLines, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  //mySweepMPIScan(resDup, numLines, MPI_INT, MPI_COMM_WORLD);
  myMPIScan(resDup, numLines);

  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  endTime = getTime();

  if (myRank == 0) {
    printf("\nTime taken for custom scan: %1f", endTime - startTime);
    //printf("\nDisplaying results of custom scan :");
  }
  fflush(stdout);
  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  //odisplayArr(resDup, numLines);
  fflush(stdout);
  MPI_Finalize();
  
  //compare both results
  i = memcmp(res, resDup, numLines*sizeof(int));
  
  if (i == 0) {
    printf("\nrank: %d both results are same\n", myRank);
  } else {
    printf("\nrank: %d both results are not same\n", myRank);
  }

  //free allocate memory
  free(nums);
  free(res);
  free(resDup);

  return 0;
  
}
