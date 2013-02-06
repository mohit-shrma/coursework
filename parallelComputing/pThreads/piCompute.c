/* code to generate pi value by randomly generating points lying in a unit 
 * square, then count the no. of points lying in the circle of diameter one
 * inside it app.  = (pi*(1/2)^2) / (1^2) = 1/4*pi. Hence we will get pi by
 * multiplying with 4 the ratio of points insde circle to the total points
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_THREADS 512
//RS_SCALE to multiply with random value to make sure it lies in between 0 and 1.0
#define RS_SCALE (1.0 / (1.0 + RAND_MAX))

void *computePi(void *);

int totalHits, totalMisses, hits[MAX_THREADS], samplePoints,
  samplePointsPerThread, numThreads;

int main(int argc, char **argv) {

  int i;
  pthread_t p_threads[MAX_THREADS];
  pthread_attr_t attr;
  double computedPi;
  double timeStart, timeEnd;
  int temp;
  struct timeval tv;
  struct timezone tz;
  
  pthread_attr_init(&attr);
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

  printf("Enter number of sample points: ");
  scanf("%d", &samplePoints);
  
  printf("Enter number of threads: ");
  scanf("%d", &numThreads);
  
  gettimeofday(&tv, &tz);
  timeStart = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
  
  totalHits = 0;
  samplePointsPerThread = samplePoints/numThreads;

  //create threads
  for (i=0; i < numThreads; i++) {
    hits[i] = i;
    //note that here hits[i] will act as seed for random no. generation
    //when threads complete they will write hit counts in the same location
    temp = pthread_create(&p_threads[i], &attr, computePi, (void *) &hits[i]);
  }

  //wait for threads to complete
  for (i=0; i < numThreads; i++) {
    temp = pthread_join(p_threads[i], NULL);
    //count the hits for each thread
    totalHits += hits[i];
  }

  //pi value computation = 4*points in circle/total points in square
  computedPi = 4.0 * (double) totalHits/((double)samplePoints);


  gettimeofday(&tv, &tz);
  timeEnd =  (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
  
  printf("Computed PI = %1f\n", computedPi);
  printf("Time taken: %1f\n", timeEnd - timeStart);

  return 0;
}


void *computePi(void *s) {
  int seed, i, *hitPointer;
  double randX, randY, randNoX, randNoY, xDiff, yDiff;
  int localHits;

  hitPointer = (int *)s;
  seed = *hitPointer;
  localHits = 0;

  for (i = 0; i < samplePointsPerThread; i++) {
    //generate a random point or coordinate
    randX = (double) (rand_r(&seed));
    randY = (double) (rand_r(&seed));
    //scale the point to make it lie in between 0 and 1.0
    randNoX = randX * RS_SCALE; //(double) ((2<<14) - 1);
    randNoY = randY * RS_SCALE; //(double) ((2<<14) - 1);
    //Following dont work as rand_r is returning much bigger number than (2<<14)-1
    //randNoX = randX / (double) ((2<<14) - 1);
    //randNoY = randY / (double) ((2<<14) - 1);
    xDiff = (randNoX - 0.5);
    yDiff = (randNoY - 0.5);
    if ((xDiff*xDiff + yDiff*yDiff) < 0.25) {
      // random point lies in the circle inside square
      localHits++;
    }
    seed *= i;
  }

  *hitPointer = localHits;
  pthread_exit(0);
  
} 
