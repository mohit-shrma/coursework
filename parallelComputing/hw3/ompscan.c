/*
 * this program will implement scan in parallel using openmp
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


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
  if (numCount %1 == 0) {
    //array is even sized
    *arr = (int *) malloc(sizeof(int) * numCount);
  } else {
    //add one element in end to make array even sized
    *arr = (int *) malloc(sizeof(int) * (numCount+1));
    //initialize the lat elem to be 0
    *arr[numCount] = 0;
  }

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




void OMP_Scan(int *arr, int arrLen, int numThreads) {
  
  int d, k, modArrLen;
  int temp, tempPow;

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

  printf("\nOriginal array: ");
  displayArr(nums, numCount);
  
  //apply the scan on i/p array
  OMP_Scan(nums, numCount, numThreads);

  if (opFileName) {
    //save the output
    writeOp(nums, numCount, opFileName);
  }

  free(nums);
  
  return 0;
}
