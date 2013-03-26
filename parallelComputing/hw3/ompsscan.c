/*
 * this program will implement scan in parallel using openmp
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

//RS_SCALE to multiply with random value to make sure it lies in between 0 and 1.0
#define RS_SCALE (1.0 / (1.0 + RAND_MAX))

struct timeval tv;
struct timeval tz;

double getTime() {
  double currTime;
  gettimeofday(&tv, &tz);
  currTime = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
  return currTime;
}

//read the numbers from file and return the count of nnumbers
int readNums(char *fileName, int **arr) {
  //to store number count
  size_t numCount;
  int i;
  FILE *fin;

  if ((fin = fopen(fileName, "r")) == NULL) {
    fprintf(stderr, "Error: failed to open '%s' for reading\n", fileName);
    exit(1);
  } else if (fscanf(fin, "%zu\n", &numCount) != 1) {
    fprintf(stderr, "Error: failed to read first line of '%s' \n", fileName);
    fclose(fin);
    exit(2);
  }
  
  //allocate space numlines numbers
  *arr = (int *) malloc(sizeof(int) * numCount);


  for (i = 0; i < numCount; i++) {
    if (fscanf(fin, "%d\n", (*arr)+i) != 1) {
      fprintf(stderr, "Err: failed to read integer from line %zu/%zu in '%s'\n",\
	      i, numCount, fileName);
      fclose(fin);
      exit(3);
    }
  }
  
  fclose(fin);
  return numCount;
}


//routinr to display array rank by rank
void displayArr(int *arr, int arrLen) {
  
  int i;

  //display the results
  for ( i = 0; i < arrLen; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}


//check if the passed floating point number is integer
int isInteger(float num) {
  float diff = num - (int)num;
  if (diff == 0) {
    return 1;
  }
  return 0;
}


int isPowerOf2(int num) {
  return num && !(num & (num-1));
}


//perform shifting for inclusive scan
void shift(int *arr, int arrLen, int last) {
  int k;
  for (k = 0; k < arrLen-1; k++) {
    arr[k] = arr[k+1];
  }
  //add the last element in the end
  arr[arrLen - 1] = last + arr[arrLen-2];
}


//perform shifting for inclusive scan
void shiftNP2(int *arr, int arrLen, int last) {
  int k;
  for (k = 0; k < arrLen-1; k++) {
    arr[k] = arr[k+1];
  }
  //add the last element in the end
  arr[arrLen - 1] = last;
}



//insert passed element into set and return 0 if full set used
int insertInSet(int *set, int setSize, int num) {
  int i;

  for (i = 0; i < setSize; i++) {
    if (set[i] == -1 || set[i] == num) {
      set[i] = num;
      break;
    }
  }

  return (setSize - 1 - i);
}



//create sqrt(n) flags to be set in array
int *createSegmentedFlags(int arrLen) {

  int seed, i, setSize;
  int *flags, *set;
  double r;

  //seed for random number
  seed = arrLen;
  
  //allocate space for set to  track which index allocated
  setSize = (int)sqrt(arrLen);
  set = (int *) malloc(sizeof(int) * setSize);
  for (i = 0; i < setSize; i++) {
    set[i] = -1;
  }

  //allocate space for flags
  flags = (int *)malloc(sizeof(int)*arrLen);
  memset(flags, 0, arrLen*sizeof(int));
  
  //generate a random number between 0 and arrLen 
  r = ((double) rand_r(&seed)) * RS_SCALE;
  r = r * arrLen;
  flags[(int)r] = 1;
  while (insertInSet(set, setSize, (int)r)) {
    r = ((double) rand_r(&seed)) * RS_SCALE;
    r = r * arrLen;
    flags[(int)r] = 1;
  }

  free(set);

  return flags;
}



//perform shift for segmented scan
void segmentShift(int *arr, int *origArr, int *flags, int arrLen) {
  int i;

  for (i = 0; i < arrLen - 1; i++) {
    if (flags[i+1] == 1) {
      //end of current segment, pull out the last element of segment
      arr[i] = arr[i] + origArr[i]; 
    } else {
      //shift the next to current pos
      arr[i] = arr[i+1];
    }
  }


  //last element is flagged, place last element of orig array
  if (flags[arrLen - 1]) {
    arr[arrLen - 1] = origArr[arrLen - 1];
  } else {
    arr[arrLen - 1] = origArr[arrLen - 1] + arr[arrLen - 2];
  }

}




//implement parallel version of segmented scan using up and down sweep
void ompSScan(int *arr, int arrLen, int *flags, int numThreads) {
  int d, k;
  int temp, tempPow;
  int last;
  int *dupFlags, *dupArr;

  assert(isPowerOf2(arrLen));

  //create copy of original flags
  dupFlags = (int *) malloc(sizeof(int)*arrLen);
  memcpy(dupFlags, flags, sizeof(int)*arrLen);

  //create copy of original array
  dupArr = (int *) malloc(sizeof(int)*arrLen);
  memcpy(dupArr, arr, sizeof(int)*arrLen);

  /*printf("\nBefore up-down sweep: ");
  displayArr(arr, arrLen);
  printf("\n");*/
  //last element of array
  last = arr[arrLen - 1];

  //perform up-sweep on the arr
  for(d = 0; d <= (int)log2(arrLen) - 1; d++) {
    //for the current level get partial sums
    tempPow = (int)pow(2, d+1);
    /*
    printf("\nd:%d tempPow:%d\n", d, tempPow);
    displayArr(arr, arrLen);
    printf("\n");
    displayArr(flags, arrLen);
    */
#pragma omp parallel for default(none)			\
  shared(arrLen, d, arr, tempPow, flags) private(k)	\
  num_threads(numThreads)
    for (k = 0; k < arrLen; k += tempPow) {
      if (!flags[k + tempPow - 1]) {
	//if flag not set then do the sum
	//printf("\nk:%d d:%d accessing: %d", k, d, k + tempPow - 1);
	arr[k + tempPow - 1] = arr[k + tempPow - 1]	\
	  + arr[k + (int)(pow(2, d)) - 1];
      }
      //or the flags to move up the flag
      flags[k + tempPow - 1] = flags[k + (int)(pow(2, d)) - 1]\
	| flags[k + tempPow - 1];
    }
  }

  /*printf("\nAfter up sweep: ");
  displayArr(arr, arrLen);
  printf("\n");
  displayArr(flags, arrLen);*/
  //perform down-sweep on array
  arr[arrLen - 1] = 0;
  for (d = ((int)log2(arrLen) - 1); d >= 0; d--) {
    tempPow = (int)pow(2, d+1);
    
    /*printf("\nd:%d tempPow:%d\n", d, tempPow);
    displayArr(arr, arrLen);
    printf("\n");
    displayArr(flags, arrLen);*/
    
    //for the level propagate the reductions
#pragma omp parallel for default(none)			\
  shared(arrLen, d, arr, tempPow, flags) private(k, temp)	\
  num_threads(numThreads)
    for (k = 0; k < arrLen; k += tempPow) {
      //printf("\nk:%d d:%d accessing & set temp to at %d", k, d, k + (int)pow(2, d) - 1);
      temp = arr[k + (int)pow(2, d) - 1];
      //printf("\nk:%d d:%d set at: %d to at %d", k, d, k + (int)(pow(2, d)) - 1, k + tempPow - 1);
      arr[k + (int)(pow(2, d)) - 1] = arr[k + tempPow - 1];
      
      if (flags[k + (int)pow(2, d)] && d) {
	//printf("\nk:%d d:%d accessing and set 0: %d", k, d, k + tempPow - 1);
	arr[k + tempPow - 1] = 0;
      } else if (flags[k + (int)pow(2, d) - 1]) {
	//printf("\nk:%d d:%d accessing %d and set %d", k, d, k + tempPow - 1, temp);
	arr[k + tempPow - 1] = temp;
      } else {
	//printf("\nk:%d d:%d accessing %d and add %d", k, d, k + tempPow - 1, temp);
	arr[k + tempPow - 1] = temp + arr[k + tempPow - 1];
      }
      //unset flag
      //printf("\nk:%d d:%d unset flag at %d", k, d, k + (int)pow(2, d) - 1);
      flags[k + (int)pow(2, d) - 1] = 0;

    }
  }

  /*printf("\nAfter down sweep: ");
  displayArr(arr, arrLen);
  printf("\n");
  displayArr(flags, arrLen);*/
  //do inclusive scan by shifting each element one index before
  segmentShift(arr, dupArr, dupFlags, arrLen);

  //printf("\nAfter shifting for inclusive scan: ");
  //displayArr(arr, arrLen);

  free(dupFlags);
  free(dupArr);
}



//implement parallel version of segmented scan using up and down sweep for non-2 power
void ompNP2SScan(int *arr, int arrLen, int *flags, int numThreads) {
  int d, k;
  int temp, tempPow;
  int last;

  int virtualRootIndex, virtualRootVal, virtualSize;
  int *virtualFlags, *dupArr;


  //create copy of original array
  dupArr = (int *) malloc(sizeof(int)*arrLen);
  memcpy(dupArr, arr, sizeof(int)*arrLen);

  temp = (int) log2(arrLen);
  virtualSize = (int) pow(2, temp + 1);
  //TODO: free virtual flags
  virtualFlags =  (int *) malloc(sizeof(int)*virtualSize);
  memset(virtualFlags, 0, sizeof(int) * virtualSize);
  memcpy(virtualFlags, flags, sizeof(int) * virtualSize);

  //printf("\nBefore up-down sweep: ");
  //displayArr(arr, arrLen);

  //last element of array
  last = -1;
  virtualRootVal = 0;
  virtualRootIndex = virtualSize - 1;
  //perform up-sweep on the arr
  for(d = 0; d <= (int)log2(virtualSize) - 1; d++) {
    //for the current level get partial sums
    tempPow = (int)pow(2, d+1);
    #pragma omp parallel for default(none)				\
  shared(d, arr, tempPow, arrLen,					\
	 virtualRootVal, virtualRootIndex, virtualSize, virtualFlags) private(k)	\
	 num_threads(numThreads)
    for (k = 0; k < virtualSize; k += tempPow) {
      if (!virtualFlags[k + tempPow - 1]) {
	if ((k + (int)(pow(2, d)) - 1) > arrLen-1			\
	    && virtualRootIndex != (k + (int)(pow(2, d)) - 1)) {
	  //dont do anything pair dont exists
	} else if (virtualRootIndex == (k + (int)(pow(2, d)) - 1)) {
	  virtualRootIndex = k + tempPow - 1;
	}else if (k + tempPow - 1 > arrLen - 1){
	  virtualRootVal += arr[k + (int)(pow(2, d)) - 1];
	  virtualRootIndex = k + tempPow - 1;
	} else {
	  arr[k + tempPow - 1] = arr[k + tempPow - 1]	\
	    + arr[k + (int)(pow(2, d)) - 1];
	}
      }
      //or the virtualFlags to move up the flag
      virtualFlags[k + tempPow - 1] = virtualFlags[k + (int)(pow(2, d)) - 1]\
	| virtualFlags[k + tempPow - 1];

    }
  }

  /*printf("\nAfter up sweep: ");
  displayArr(arr, arrLen);
  printf("\nvirtualRootIndex=%d", virtualRootIndex);
  printf("\nvirtualRootVal=%d", virtualRootVal);
  */
  //perform down-sweep on array
  virtualRootVal = 0;
  
  //propagate the reductions
  for (d = ((int)log2(virtualSize) - 1); d >= 0; d--) {
    tempPow = (int)pow(2, d+1);
    //printf("\nd:%d tempPow:%d", d, tempPow);
    //for the level propagate the reductions
#pragma omp parallel for default(none)				\
  shared(arrLen, d, arr, tempPow, virtualRootVal,		\
	 virtualRootIndex, virtualSize, virtualFlags) private(k, temp)	\
  num_threads(numThreads)
    for (k = 0; k < virtualSize; k += tempPow) {
      if (k + tempPow - 1 < arrLen) {
	//do normal work
	temp = arr[k + (int)pow(2, d) - 1];
	//printf("\nk:%d d:%d accessing: %d", k, d, k + tempPow - 1);
	arr[k + (int)(pow(2, d)) - 1] = arr[k + tempPow - 1];

	if (virtualFlags[k + (int)pow(2, d)] && d) {
	  //printf("\nk:%d d:%d accessing and set 0: %d", k, d, k + tempPow - 1);
	  arr[k + tempPow - 1] = 0;
	} else if (virtualFlags[k + (int)pow(2, d) - 1]) {
	  //printf("\nk:%d d:%d accessing %d and set %d", k, d, k + tempPow - 1, temp);
	  arr[k + tempPow - 1] = temp;
	} else {
	  //printf("\nk:%d d:%d accessing %d and add %d", k, d, k + tempPow - 1, temp);
	  arr[k + tempPow - 1] = temp + arr[k + tempPow - 1];
	}

      } else if (k + (int)pow(2, d) - 1 > arrLen - 1		\
		 && virtualRootIndex == k + tempPow - 1) {
	virtualRootIndex = k + (int)pow(2, d) - 1;
	//virtualRootVal = virtualRootVal;
	//no need to change root
      } else if (k + (int)pow(2, d) - 1 < arrLen) {

	//left child inside, root/right child outside
	temp = arr[k + (int)pow(2, d) - 1];
	arr[k + (int)pow(2, d) - 1] = virtualRootVal;

	if (virtualFlags[k + (int)pow(2, d)] && d) {
	  //printf("\nk:%d d:%d accessing and set 0: %d", k, d, k + tempPow - 1);
	  virtualRootVal = 0;
	} else if (virtualFlags[k + (int)pow(2, d) - 1]) {
	  //printf("\nk:%d d:%d accessing %d and set %d", k, d, k + tempPow - 1, temp);
	  virtualRootVal = temp;
	} else {
	  //printf("\nk:%d d:%d accessing %d and add %d", k, d, k + tempPow - 1, temp);
	  virtualRootVal += temp;
	}

      }
      
      //unset flag
      //printf("\nk:%d d:%d unset flag at %d", k, d, k + (int)pow(2, d) - 1);
      virtualFlags[k + (int)pow(2, d) - 1] = 0;
    }
  }

  /*
  printf("\nAfter down sweep: ");
  displayArr(arr, arrLen);
  printf("\nvirtualRootIndex=%d", virtualRootIndex);
  printf("\nvirtualRootVal=%d", virtualRootVal);
  */

  //do inclusive scan by shifting each element one index before
  segmentShift(arr, dupArr, flags, arrLen);

  //printf("\nAfter shifting for inclusive scan: ");
  //displayArr(arr, arrLen);

  free(virtualFlags);
  free(dupArr);
}



//perform segmented scan on the passed array, this will call non power of 2 segmented scan
void performSScan2(int *arr, int arrLen, int numThreads) {
  int *flags;

  //get segmented flags
  flags = createSegmentedFlags(arrLen);
  //printf("\nSegmented flags...");
  //displayArr(flags, arrLen);
  
  ompNP2SScan(arr, arrLen, flags, numThreads);

  //printf("\n After performing scan: ");
  //displayArr(arr, arrLen);
}


//perform segmented scan on the passed array
void performSScan(int *arr, int arrLen, int numThreads) {
  int nearPow, i, isFlagFound;
  int *tempArr, tempArrLen;
  int numExtraElem;
  int *flags;
  int *tempFlags;
  //int flags[9] = {1, 0, 1, 0, 0, 0, 1, 0, 1};
  //int flags[8] = {0, 1, 0, 0, 0, 0, 1, 0};
  //get segmented flags
  flags = createSegmentedFlags(arrLen);
  //printf("\nSegmented flags...");
  //displayArr(flags, arrLen);

  if (isPowerOf2(arrLen)) {
    //passed array is power is power of 2, Yipee!
    ompSScan(arr, arrLen, flags, numThreads);
  } else {
    //passed array not power of 2 
    //get lower nearest power of 2
    nearPow = (int)log2(arrLen);

    //allocate int[] of length in  power of 2 <= arrLen
    tempArrLen = pow(2, nearPow); 
    numExtraElem = arrLen - tempArrLen;
    tempArr = (int*) malloc(sizeof(int)*tempArrLen);
    
    //allocate space for temporary flags
    tempFlags = (int*) malloc(sizeof(int)*tempArrLen);
    
    //copy tempArrLen flags from flags into tempFlags
    memcpy(tempFlags, flags, tempArrLen*sizeof(int));

    //copy tempArrLen number from arr into tempArr
    memcpy(tempArr, arr, tempArrLen*sizeof(int));

    //do scan of this array
    ompSScan(tempArr, tempArrLen, tempFlags, numThreads);

    //copy these values back to orig array
    memcpy(arr, tempArr, tempArrLen*sizeof(int));

    //clear tempArr to 0
    memset(tempArr, 0, tempArrLen*sizeof(int));
    memset(tempFlags, 0, tempArrLen*sizeof(int));
    
    //copy remaining values from original array
    memcpy(tempArr, arr + tempArrLen,\
	   numExtraElem*sizeof(int));
    //copy remaining flags to tempFlags
    memcpy(tempFlags, flags + tempArrLen,\
	   numExtraElem*sizeof(int));

    //apply scan on the temparray
    ompSScan(tempArr, tempArrLen, tempFlags, numThreads);

    //copy values back to temp array copying 
    //the last result of previous scan
    for (i = 0; i < numExtraElem; i++) {
      if (flags[i+tempArrLen]) {
	isFlagFound = 1;
      }
      if (!isFlagFound) {
	//add results from last segment until 1 is not found
	arr[i + tempArrLen] = tempArr[i] + arr[tempArrLen-1];
      } else {
	//1 was found in flags, just copy values
	arr[i + tempArrLen] = tempArr[i];
      }
    }
    
    //free allocated mem for tempArr
    free(tempArr);
    free(tempFlags);
  }

  //printf("\n After performing scan: ");
  //displayArr(arr, arrLen);
}



//write the output of scan to the output file
void writeOp(int *arr, int count, char *opFileName) {
  FILE *opFile;
  int i;

  opFile = fopen(opFileName, "w");
  
  for (i = 0; i < count; i++) {
    fprintf(opFile, "%d\n", arr[i]);
  }
  fclose(opFile);
}


int main(int argc, char *argv[]) {
  
  //store the array and length of array
  int *nums, numCount;
  //input and output filename containing numbers
  char *ipFileName, *opFileName;
  //number of threads
  int numThreads;
  
  double endTime, startTime;

  if (argc < 3) {
    printf("Error: insufficient args.\n");
    exit(1);
  } else {
    numThreads = atoi(argv[1]);
    ipFileName = argv[2];
    if (argc == 4) {
      opFileName = argv[3];
    } else {
      opFileName = (char *)0;
    }
  }

  //read the numbers from i/p file
  numCount = readNums(ipFileName, &nums);

  //printf("\nOriginal array: ");
  //displayArr(nums, numCount);

  startTime = getTime();
  
  //apply the scan on i/p array
  printf("\n\nPerforming segmented scan:");
  //uncomment to run scan by splittiong power of 2 group
  //performSScan(nums, numCount, numThreads);
  //scan for generic power of 2
  performSScan2(nums, numCount, numThreads);

  endTime = getTime();
  printf("\nNum threads: %d Time taken: %1f\n",\
	 numThreads, endTime - startTime);
  
  if (opFileName) {
    //save the output
    writeOp(nums, numCount, opFileName);
  }

  free(nums);
  
  return 0;
}
