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
  int *grpRanks, *dupArr;
  int i, tag, status;
  int remProcRankStart;
  int borrowRankstart;
  
  MPI_Group origGroup, newGroup;
  MPI_Comm newComm;
  
  dupArr = (int*) 0;
  tag = 100;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (isPowerOf2(numProcs)) {
    printf("\n num of procs is power of 2");
    mySweepMPIScan(arr, count, MPI_INT, MPI_COMM_WORLD);
  } else {
    //num of processes not power of 2 
    //passed num procs not power of 2 
    //get lower nearest power of 2
    nearPow = (int)log2(numProcs);
    tempNumProcs = pow(2, nearPow);

    if (myRank == 0) {
      printf("\n tempNumProcs: %d", tempNumProcs);
    }

    //extract original group handle
    MPI_Comm_group(MPI_COMM_WORLD, &origGroup);

    //create a group of near pow number of procs
    grpRanks = (int*) malloc(sizeof(int)*tempNumProcs);
    for (i = 0; i < tempNumProcs; i++) {
      // add rank i to group
      grpRanks[i] = i;
    }

    //TODO: make sure new ranks in same order as original rank
    //add procs to groups
    if (myRank <= tempNumProcs) {
      MPI_Group_incl(origGroup, tempNumProcs, grpRanks, &newGroup);
      //create a new communicator
      MPI_Comm_create(MPI_COMM_WORLD, newGroup, &newComm);
      //get new rank
      MPI_Group_rank(newGroup, &newRank);
      printf("\n first grouping Orig rank = %d New rank = %d", myRank, newRank);
      
      //perform sweep mpi scan on this group
      //mySweepMPIScan(arr, count, MPI_INT, newComm);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //temp group finished sweep scan operations
    
    //need to perform scan on remaining procs
    
    //number of processors to borrow 
    procsNeeded = numProcs - tempNumProcs;

    
    //dup the result for procs that will be borrowed
    if (myRank < procsNeeded) {
      dupArr = (int *)malloc(sizeof(int)*count);
      memcpy(dupArr, arr, sizeof(int)*count);
      //clear the array after storing result
      //TODO: not necessay as not concerned with what goes here
      memset(arr, 0, sizeof(int)*count);
    }

    
    //form the new group with borrowed procs in end
    remProcRankStart = tempNumProcs;
    borrowRankstart = 0;
    for (i = 0; i < tempNumProcs; i++) {
      if (remProcRankStart < numProcs) {
	grpRanks[i] = remProcRankStart++;
      } else {
	grpRanks[i] = borrowRankstart++;
      }
    }
    

    if (myRank == 0) {
      printf("\n Group ranks including borrowed procs: \n");
      for (i = 0; i < tempNumProcs; i++) {
	printf(" %d", grpRanks[i]);
      }
      printf("\n");
    }
    

    

    //form a new group to perform rest of scan
    if (myRank < procsNeeded || myRank >= tempNumProcs) {
      //form new group
      MPI_Group_incl(origGroup, tempNumProcs, grpRanks, &newGroup);
      
      //create a new communicator
      MPI_Comm_create(MPI_COMM_WORLD, newGroup, &newComm);

      //get new rank
      MPI_Group_rank(newGroup, &newRank);
      printf("\nSecond grouping Orig rank = %d New rank = %d", myRank, newRank);
      
      //perform sweep mpi scan on this group
      //mySweepMPIScan(arr, count, MPI_INT, newComm);
      
    }
    
    //make sure each procs reach this point
    MPI_Barrier(MPI_COMM_WORLD);
    /*
    //restore values from dupArr to arr
    if (myRank < procsNeeded) {
      memcpy(arr, dupArr, sizeof(int)*count);
    }
    
    //add the prefix scan result from tempNumProcs - 1 to remaining procs
    if (myRank == tempNumProcs - 1) {
      //send arr to all procs with rank >= tempNumProcs
      //can use bcast here
      for (i = tempNumProcs; i < numProcs; i++) {
	MPI_Send(&arr, count, MPI_INT, i, tag, MPI_COMM_WORLD);
      }
    } else if (myRank >= tempNumProcs) {
      if (!dupArr) {
	dupArr = (int *)malloc(sizeof(int)*count);
      }

      //recv arr from proc tempNumProcs - 1 in dup arr
      MPI_Recv(&dupArr, count, MPI_INT, tempNumProcs - 1, tag, MPI_COMM_WORLD,	\
	       &status);

      //add the results from received array
      for (i = 0; i < count; i++) {
	arr[i] += dupArr[i];
      }
      
    }
    */

    printf("\nbefore free mem");
    if (dupArr) {
      free(dupArr);
    }
    free(grpRanks);
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
  MPI_Status status;
  
  MPI_Comm_size(comm, &numProcs);
  MPI_Comm_rank(comm, &myRank);
  
  tag = 100;
  
  assert(isPowerOf2(numProcs));

  printf("\n Number of procs: %d", numProcs);
  printf("\n My rank: %d", myRank);
  if (isPowerOf2(numProcs)) {
    for (i = 0; i < count; i++) {
      
      if (myRank == numProcs -1) {
	//save the last proc's element as this is going to be root
	last = arr[i];
      }
      


      MPI_Barrier(comm);

      if (myRank == 0) {
	printf("\n\nstarting up-sweep phase");
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
	printf("\nstarting down-sweep phase");
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
    printf("\nDisplaying results of custom scan :");
  }
  fflush(stdout);
  //make sure every process reach this checkpoint
  MPI_Barrier(MPI_COMM_WORLD);

  displayArr(resDup, numLines);
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
