
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


//create matrix of given dimension and zero out it's values
int** createZMat(int m, int n) {
  //allocate space for m rows
  int** mat = (int**) malloc(m*sizeof(int*));
  int i;
  for (i = 0; i < m; i++) {
    //allocate space for columns
    mat[i] = (int*) malloc(n*sizeof(int));
    //zeroed values into this row of matrix
    memset(mat[i], 0, n);
  }

  return mat;
}


//zero out a matrix
void zeroed(int** a, int m, int n) {
  int i, j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      a[i][j] = 0;
    }
  }
}

//input matrix from user, m rows, n cols
int** getMatrix(int m, int n) {
  //allocate space for m rows
  int** mat = (int**) malloc(m*sizeof(int*));
  int i, j;
  for (i = 0; i < m; i++) {
    //allocate space for columns
    mat[i] = (int*) malloc(n*sizeof(int));
    //input values into this row of matrix
    for (j = 0; j < n; j++) {
      //or try &mat[i][j]
      scanf("%d", mat[i]+j);
    }
  }

  return mat;
}


//display the specified matrix
void displayMatrix(int **a, int m, int n) {
  int i, j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%d ", a[i][j]);
    }
    printf("\n");
  }
}


//compute the product of two matrices
void product(int** a, int** b, int** c, int aM, int aN, int bM, int bN) {
  //dimension of product matrix is aM*bN
  int i, j, k;
  for (i = 0; i < aM; i++) {
    for (j = 0; j < bN; j++) {
      for (k = 0; k < aN; k++) {
	c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}


//compute the product of two matrices using nested for parallely
void productParNested(int** a, int** b, int** c,\
		      int aM, int aN, int bM, int bN) {
  //dimension of product matrix is aM*bN
  int i, j, k;
  //NOTE: use of first private to retain previous values
  //otherwise program is crashing
  #pragma omp parallel for default(none) firstprivate(j, k)\
    shared(a, b, c, aM, bN, aN)	num_threads(2)
    for (i = 0; i < aM; i++) {
      #pragma omp parallel for default(none) firstprivate(i, k)\
        shared(a, b, c, aM, bN, aN) num_threads(2)
       for (j = 0; j < bN; j++) {
	 c[i][j] = 0;
         #pragma omp parallel for default(none) firstprivate(i, j)	\
           shared(a, b, c, aM, bN, aN)	num_threads(2)
	 for (k = 0; k < aN; k++) {
	   c[i][j] += a[i][k] * b[k][j];
	 } 
       }
    }
}



//compute the product of two matrix parallely
void productPar(int** a, int** b, int** c, int aM, int aN, int bM, int bN) {
  //dimension of product matrix is aM*bN
  int i, j, k;
  #pragma omp parallel default(none) private(i, j, k) shared(a, b, c, aM, bN, aN)\
    num_threads(4)
    #pragma omp for
    for (i = 0; i < aM; i++) {
      /*#pragma omp for can't do this
       *warning: work-sharing region may not be closely nested inside of
       *work-sharing, critical, ordered or master region*/
      for (j = 0; j < bN; j++) {
	for (k = 0; k < aN; k++) {
	  c[i][j] += a[i][k] * b[k][j];
	}
      }
    }
  
}


int main(int argc, char **argv) {

  //initialize pointer to two matrices
  int **a, **b;
  
  //initialize pointer to third matrix
  int **c;
  
  //dimension of matrices, M rows & N cols
  int aM, aN, bM, bN;

  printf("Enter no. of rows of first matrix: ");
  scanf("%d", &aM);

  printf("Enter no. of cols of first matrix: ");
  scanf("%d", &aN);
  
  printf("Enter no. of rows of second matrix: ");
  scanf("%d", &bM);

  printf("Enter no. of cols of second matrix: ");
  scanf("%d", &bN);

  if (aN != bM) {
    //columns of first not equal to rows of second
    printf("\n Input matrices can't be multiplied.\n");
  } else {
    //get values of matrix a
    printf("Enter values of matrix A: \n");
    a = getMatrix(aM, aN);
    
    //get values of matrix b
    printf("Enter values of matrix B: \n");
    b = getMatrix(bM, bN);

    //show matrix A
    printf("Matrix A: \n");
    displayMatrix(a, aM, aN);
    
    //show matrix B
    printf("Matrix B: \n");
    displayMatrix(b, bM, bN);

    //get zeroed matrix C
    c = createZMat(aM, bN);
    //displayMatrix(c, aM, bN);
    //compute product of matrices
    printf("Product of two matrix is as follow (serial): \n");
    product(a, b, c, aM,  aN,  bM,  bN);
    displayMatrix(c, aM, bN);
    zeroed(c, aM, bN);
    
    printf("Product of two matrix is as follow (parallel for): \n");
    productPar(a, b, c, aM,  aN,  bM,  bN);
    displayMatrix(c, aM, bN);
    zeroed(c, aM, bN);
    
    printf("Product of two matrix is as follow (parallel nested for): \n");
    productParNested(a, b, c, aM,  aN,  bM,  bN);
    displayMatrix(c, aM, bN);
    zeroed(c, aM, bN);
  }

  free(a);
  free(b);
  free(c);

  return 0;
}
