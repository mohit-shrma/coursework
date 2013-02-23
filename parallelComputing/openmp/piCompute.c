/* code to generate pi value by randomly generating points lying in a unit 
 * square, then count the no. of points lying in the circle of diameter one
 * inside it app.  = (pi*(1/2)^2) / (1^2) = 1/4*pi. Hence we will get pi by
 * multiplying with 4 the ratio of points insde circle to the total points
 * will use openmp framework here, compile with -fopenmp
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

//RS_SCALE to multiply with random value to make sure it lies in between 0 and 1.0
#define RS_SCALE (1.0 / (1.0 + RAND_MAX))

int main(int argc, char **argv) {

  int samplePoints, numThreads, sum;
  double computedPi;
  printf("Enter number of sample points: ");
  scanf("%d", &samplePoints);
  
  printf("Enter number of threads: ");
  scanf("%d", &numThreads);

  double randX, randNoX, randY, randNoY;
  double xDiff, yDiff;
  unsigned int seed, samplePointsPerThread;
  int i, threadId;

  //Not initializing to 0 don't give good results looks like value afer
  //reduction is added here
  sum = 0;

  /*start openmp parallel block
   *by default all values are private that is copied across thread
   *single copy of samplePoints is used
   *after the completion '+' is applied as reduction operatoe i.e after exiting
   *parallel block 'sum' contains sum of all the copies of variable 'sum'  
   */
   #pragma omp parallel default(none) private(numThreads, threadId, seed, \
	   samplePointsPerThread, randX, randY, randNoX, randNoY, xDiff,\
	   yDiff, i) shared(samplePoints) reduction(+: sum) \
           num_threads(numThreads)
  {
    //TODO: whats the diff b/w assignment here and pragma above
    numThreads = omp_get_num_threads();

    //get thread id to seed random number generation
    threadId = omp_get_thread_num();
    seed = threadId;
    
    //no. of sample points to be generated per thread
    samplePointsPerThread = samplePoints/numThreads;
    //sum will store the no. of points falling in circle
    sum = 0;
    //printf("\nsample points per thread: %d", samplePointsPerThread);
    /*following pragma will split the following for loop across threads, 
      loop index goes to total number of points not to num pointes per thread*/
    #pragma omp for 
    for (i = 0; i < samplePoints; i++) {
    //for (i = 0; i < samplePointsPerThread; i++) { //comment for pragma and uncomment this
      //generate a random point or coordinate
      randX = (double) (rand_r(&seed));
      randY = (double) (rand_r(&seed));
      //scale the point to make it lie in between 0 and 1.0
      randNoX = randX * RS_SCALE; //(double) ((2<<14) - 1);
      randNoY = randY * RS_SCALE; //(double) ((2<<14) - 1);
      xDiff = (randNoX - 0.5);
      yDiff = (randNoY - 0.5);
      if ((xDiff*xDiff + yDiff*yDiff) < 0.25) {
	// random point lies in the circle inside square
	sum++;
      }
      //TODO: don't know how to multiply seed with i when using prgma omp for
      //seed *= i;
    }
  }

  //after applying reduction sum will contain total number of points in circle
  //pi value computation = 4*points in circle/total points in square
  printf("\nhits = %d", sum);
  printf("\nsample points = %d", samplePoints);
  computedPi = 4.0 * (double)sum / (double)samplePoints;
  printf("\nComputed PI = %1f\n", computedPi);

  return 0;
}
