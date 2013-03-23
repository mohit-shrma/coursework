/*
 * this program will implement scan in parallel using openmp
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
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



//implement serial version of scan using up and down sweep
void ompScan(int *arr, int arrLen, int numThreads) {
  int d, k;
  int temp, tempPow;
  int last;

  assert(isPowerOf2(arrLen));

  //last element of array
  last = arr[arrLen - 1];

  //perform up-sweep on the arr
  for(d = 0; d <= (int)log2(arrLen) - 1; d++) {
    //for the current level get partial sums
    tempPow = (int)pow(2, d+1);
#pragma omp parallel for default(none)		\
  shared(arrLen, d, arr, tempPow) private(k)	\
  num_threads(numThreads)
    for (k = 0; k < arrLen; k += tempPow) {
      arr[k + tempPow - 1] = arr[k + tempPow - 1]	\
	+ arr[k + (int)(pow(2, d)) - 1];
    }
  }

  printf("\nAfter up sweep: ");
  displayArr(arr, arrLen);

  //perform down-sweep on array
  arr[arrLen - 1] = 0;
  for (d = (int)log2(arrLen); d >= 0; d--) {
    tempPow = (int)pow(2, d+1);
    //for the level propagate the reductions
#pragma omp parallel for default(none)			\
  shared(arrLen, d, arr, tempPow) private(k, temp)	\
  num_threads(numThreads)
    for (k = 0; k < arrLen; k += tempPow) {
      temp = arr[k + (int)pow(2, d) - 1];
      arr[k + (int)(pow(2, d)) - 1] = arr[k + tempPow - 1];
      arr[k + tempPow - 1] += temp; 
    }
  }

  printf("\nAfter down sweep: ");
  displayArr(arr, arrLen);

  //do inclusive scan by shifting each element one index before
  shift(arr, arrLen, last);

  printf("\nAfter shifting for inclusive scan: ");
  displayArr(arr, arrLen);
}


void OMP_Scan(int *arr, int arrLen, int numThreads) {
  
  int d, k, modArrLen;
  int temp, tempPow;
  int last;
  
  last = arr[arrLen-1];
  
  //to take care of case when array contains odd elements
  modArrLen = arrLen;
  if (arrLen %2 != 0) {
    //for odd size we have already allocate an extra elem
    //in end initialize to zero
    modArrLen += 1;
  }
  
  //perform up-sweep on the arr
  for(d = 0; d <= (int)log2(modArrLen) - 1; d++) {
    //for the current level get partial sums in parallel
    tempPow = (int)pow(2, d+1);
#pragma omp parallel for default(none) shared(modArrLen, d, arr, tempPow) private(k) \
  num_threads(numThreads)
    for (k = 0; k < modArrLen; k += tempPow) {
      arr[k + (int)(pow(2, d+1)) - 1] += arr[k + (int)(pow(2, d)) - 1];
    }
  }

  printf("\nAfter up sweep: ");
  displayArr(arr, arrLen);

  //perform down-sweep on array
  arr[modArrLen - 1] = 0;
  for (d = (int)log2(modArrLen); d >= 0; d--) {
    tempPow = (int)pow(2, d+1);
    //for the level propage the reductions in parallel
#pragma omp parallel for default(none) shared(modArrLen, d, arr, tempPow) private(k, temp) \
  num_threads(numThreads)
    for (k = 0; k < modArrLen; k += tempPow) {
      temp = arr[k + (int)pow(2, d) - 1];
      arr[k + (int)(pow(2, d)) - 1] = arr[k + (int)(pow(2, d+1)) - 1];
      arr[k + (int)(pow(2, d+1)) - 1] += temp; 
    }
  }

  printf("\nAfter down sweep: ");
  displayArr(arr, arrLen);

  //do inclusive scan by shifting each element one index before
  shift(arr, arrLen, last);

  printf("\nAfter shifting for inclusive scan: ");
  displayArr(arr, arrLen);
}



//implement serial version of scan using up and down sweep
void serialScan(int *arr, int arrLen) {
  int d, k;
  int temp, tempPow;
  int last;

  assert(isPowerOf2(arrLen));

  //last element of array
  last = arr[arrLen - 1];

  //perform up-sweep on the arr
  for(d = 0; d <= (int)log2(arrLen) - 1; d++) {
    //for the current level get partial sums
    tempPow = (int)pow(2, d+1);
    for (k = 0; k < arrLen; k += tempPow) {
	arr[k + tempPow - 1] = arr[k + tempPow - 1] \
	  + arr[k + (int)(pow(2, d)) - 1];
    }
  }

  printf("\nAfter up sweep: ");
  displayArr(arr, arrLen);

  //perform down-sweep on array
  arr[arrLen - 1] = 0;
  for (d = (int)log2(arrLen); d >= 0; d--) {
    tempPow = (int)pow(2, d+1);
    //for the level propage the reductions
    for (k = 0; k < arrLen; k += tempPow) {
	temp = arr[k + (int)pow(2, d) - 1];
	arr[k + (int)(pow(2, d)) - 1] = arr[k + tempPow - 1];
	arr[k + tempPow - 1] += temp; 
    }
  }

  printf("\nAfter down sweep: ");
  displayArr(arr, arrLen);

  //do inclusive scan by shifting each element one index before
  shift(arr, arrLen, last);

  printf("\nAfter shifting for inclusive scan: ");
  displayArr(arr, arrLen);
}





void performScan(int *arr, int arrLen, int numThreads) {
  
  int nearPow, i;
  int *tempArr, tempArrLen;
  int numExtraElem;

  if (isPowerOf2(arrLen)) {
    //passed array is power is power of 2, Yipee!
    serialScan(arr, arrLen);
  } else {
    //passed array not power of 2 
    //get lower nearest power of 2
    nearPow = (int)log2(arrLen);

    //allocate int[] of length in  power of 2 <= arrLen
    tempArrLen = pow(2, nearPow); 
    numExtraElem = arrLen - tempArrLen;
    tempArr = (int*) malloc(sizeof(int)*tempArrLen);
    
    //copy tempArrLen number from arr into tempArr
    memcpy(tempArr, arr, tempArrLen*sizeof(int));

    //do scan of this array
    //serialScan(tempArr, tempArrLen);
    OMP_Scan(tempArr, tempArrLen, numThreads);

    //copy these values back to orig array
    memcpy(arr, tempArr, tempArrLen*sizeof(int));

    //clear tempArr to 0
    memset(tempArr, 0, tempArrLen*sizeof(int));
    
    //copy remaining values from original array
    memcpy(tempArr, arr+tempArrLen,\
	   numExtraElem*sizeof(int));

    //apply scan on the temparray
    //serialScan(tempArr, tempArrLen);
    OMP_Scan(tempArr, tempArrLen, numThreads);

    //copy values back to temp array copying 
    //the last result of previous scan
    for (i = 0; i < numExtraElem; i++) {
      arr[i + tempArrLen] = tempArr[i] + arr[tempArrLen-1];
    }
    
    //free allocated mem for tempArr
    free(tempArr);
  }

  printf("\n After performing scan: ");
  displayArr(arr, arrLen);
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
  //OMP_Scan(nums, numCount, numThreads);

  performScan(nums, numCount, numThreads);

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
