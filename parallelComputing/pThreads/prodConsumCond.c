/*code for producer consumer using mutex with conditional variables
 *here is queue is assumed to be of unit size 
 */
//TODO: working only for equal consumer and producer threads only
// a way to notify all threads to stop
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_THREADS 512

pthread_cond_t condQEmpty, condQFull;
pthread_mutex_t taskQCondLock;
int taskAvailable, tasksCount, tasksFinished;

void* producer(void*);
void* consumer(void*);

int main(int argc, char **argv) {
  int i;
  int numThreads;
  int* data = (int*) 0;
  pthread_t pThreads[MAX_THREADS];
  pthread_attr_t attr;
  
  taskAvailable = 0;
  
  printf("Enter number of threads: ");
  scanf("%d", &numThreads);

  printf("Enter number of tasks: ");
  scanf("%d", &tasksCount);

  //TODO: check pthread_init();
  pthread_cond_init(&condQEmpty, NULL);
  pthread_cond_init(&condQFull, NULL);
  pthread_mutex_init(&taskQCondLock, NULL);

  pthread_attr_init(&attr);
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

  //allocate data as number of threads
  data = (int*)malloc(numThreads * sizeof(int));
  //zero out the allocated mem
  memset(data, 0, numThreads * sizeof(int));


  //create producer/consumer threads
  for (i = 0; i < numThreads; i++) {
    if (i%2 == 0) {
      //create producer thread
      pthread_create(&pThreads[i], &attr, producer, data + i);
    } else {
      //create consumer thread
      pthread_create(&pThreads[i], &attr, consumer, data + i);
    }
  }

  //join threads
  for (i = 0; i < numThreads; i++) {
    pthread_join(pThreads[i], NULL);
  }

  printf("\nTasks finished count: %d\n", tasksFinished);
  
  return 0;
}



//return non zero if all tasks finished
int done() {
  if (tasksFinished >= tasksCount) {
    return 1;
  } else {
    return 0;
  }
}



//producer implementation
void* producer(void* producerThreadData) {
  //TODO:check its use
  //int inserted;
  //while all taks are not finished
  while (!done()) {
    //createTask()
    //try to get the mutex lock
    pthread_mutex_lock(&taskQCondLock);

    while (taskAvailable == 1 && !done()) {
      //if a task is available, check to see if queue is empty using
      //conditional var, if not empty then wait for signal
      pthread_cond_wait(&condQEmpty, &taskQCondLock);
    }
    
    //insert task into queue
    //insertIntoQ()
    if (taskAvailable == 0 && !done()) { 
      printf(" \n task inserted... \n");

    }

    taskAvailable = 1;
    
    //signal consumer that queue is full
    pthread_cond_signal(&condQFull);
    //broadcast to all threads waiting on conditional variable
    //pthread_cond_broadcast(&condQFull);
    
    //release the lock on mutex
    pthread_mutex_unlock(&taskQCondLock);
  }
  return (void*)0;
}



//consumer implementation
void* consumer(void* consumerThreadData) {
  //while all taks are not finished
  while(!done()) {
    pthread_mutex_lock(&taskQCondLock);
    while (taskAvailable == 0 && !done()) {
      //if task is not available, check if task in Q using conditional
      //var, if empty then wait for signal from producer
      pthread_cond_wait(&condQFull, &taskQCondLock);
    }

    //extract task from Q
    if (taskAvailable > 0 && !done()) {
      printf(" \n task consumed... \n");
      tasksFinished++;

    } 

    taskAvailable = 0;
    
    //signal producer that Q is empty
    pthread_cond_signal(&condQEmpty);
    //broadcast to all threads waiting on conditional variable
    //pthread_cond_broadcast(&condQEmpty);
    
    //release the lock on mutex
    pthread_mutex_unlock(&taskQCondLock);

    //send task to be processed
    //processTask(extractedTask)
  }
  return (void*)0;
}
