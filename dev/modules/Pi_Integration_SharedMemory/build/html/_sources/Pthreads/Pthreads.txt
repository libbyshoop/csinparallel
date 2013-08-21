=========================================
Pi Using Numerical Integration: Pthreads
=========================================

An implementation of the area computation with the POSIX threads (Pthreads) explicit threading model is shown here. In the main() routine, a number (NUMTHREADS) of threads are spawned to execute the function Area(). This function takes one argument: (a pointer to) the thread number generated and stored in the tNum array. After the child threads are launched, the main() thread will call pthread_join to wait for each thread, in turn, to complete computation. The computed area of the half circle will then be stored in gPi. Multiply this result by 2.0 to compute the approximation to pi.

The threaded function Area() uses the thread number (0..NUMTHREADS-1) to initialize the local loop iteration variable. This value is used to compute the midpoint of a rectangle, the height of the rectangle, and then the area of the rectangle. Notice how the increment value in the for-loop is the number of threads. In the code given, this will have the loop of the first thread (myNum == 0) take on values 0, 4, 8, 12, etc., while the last thread (myNum == 3) will use the iteration values 3, 7, 11, 15, etc. This scheme ensures that all values in the NUM_RECT range are used and only used by one thread.

Rather than update the shared summation variable, gPi, each time a new rectangle area is computed, a local partial sum variable is used within each thread. Once the loop has completed, each partial sum is added to the shared sum with a critical region protected by the mutex object gLock. In this way, protected updates to the shared variable are done only once per thread (4 times) rather than once per rectangle (NUM_RECT times). ::

 #include <stdio.h>
 #include <math.h>
 #include <pthread.h>
 #define NUM_RECT 10000000
 #define NUMTHREADS 4
 double gPi = 0.0;  //  global accumulator for areas computed
 pthread_mutex_t gLock;

 void *Area(void *pArg){
            int myNum = *((int *)pArg);
            double h = 2.0 / NUM_RECT;
            double partialSum = 0.0, x;  // local to each thread

            // use every NUMTHREADS-th ste
            for (int i = myNum; i < NUM_RECT; i += NUMTHREADS){
    	x = -1 + (i + 0.5f) * h;
    	            partialSum += sqrt(1.0 - x*x) * h; 
            }
            pthread_mutex_lock(&gLock);
            gPi += partialSum;  // add partial to global final answer
            pthread_mutex_unlock(&gLock);
            return 0;
 }

 int main(int argc, char **argv) {

 pthread_t tHandles[NUMTHREADS]; int tNum[NUMTHREADS], i;
 pthread_mutex_init(&gLock, NULL);
 for ( i = 0; i < NUMTHREADS; ++i ) {
 tNum[i] = i;
 pthread_create(&tHandles[i],  	  	// Returned thread handle
      	         	NULL,                 // Thread Attributes
      	         	Area,  	  	  	// Thread function
      	         	(void)&tNum[i]);  	// Data for Area()
 }
 for ( i = 0; i < NUMTHREADS; ++i ) {
    pthread_join(tHandles[i], NULL);
 }
 gPi *= 2.0;
 printf("Computed value of Pi:  %12.9f\n", gPi );
 pthread_mutex_destroy(&gLock);
 return 0;
 }

