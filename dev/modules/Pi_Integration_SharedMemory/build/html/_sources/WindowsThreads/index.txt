================================================
Pi Using Numerical Integration: Windows Threads
================================================

An implementation of the area computation with the Windows threads (Win32 threads) explicit threading model is shown here. There is not much difference between this version and the Pthreads version. Both spawn threads, assign those threads some portion of the rectangles, compute and sum the rectangle areas, and update the shared summation variable. The main thread blocks until all threads have terminated (via WaitForMultipleObjects()).

The difference in algorithms used is how the rectangles are distributed to threads. In this version, the beginning and ending index values of the original iteration range are computed within each thread. These begin and end indices are used as for-loop bounds. The number of rectangles each thread will handle is computed by dividing the number of rectangles by the number of threads; the index values are found by multiplying this ratio by the thread number and the thread number + 1. For 1000 rectangles and four threads, the first thread (myNum == 0) will start with rectangle 0 and finish with rectangle 249 (< (myNum+1) * (1000/4)) for a total of 250 rectangles for each thread.
 
The one caveat that must be addressed with this method of dividing loop iterations is when the number of rectangles (NUM_RECT) is not divisible by the number of threads (NUMTHREADS). For example, if the number of rectangles to use were 10000019 (a prime number), dividing by any number of threads will leave some iterations left out of the computation when computing the iteration bounds as described above. For instance, if executing on 4 threads, three iterations would remain unattached to a thread. Thus, to account for any “leftover” iterations, the remainder are added to the last thread by setting the end variable to be the explicit number of rectangles. If the time to compute a single iteration is significant, this distribution scheme could lead to load imbalance and an alternate method of iteration assignment should be used. ::

	#include <windows.h>
	#include <stdio.h>
	#include <math.h>
 
	#define NUM_RECT 10000000
	#define NUMTHREADS 4
	 
	double gPi = 0.0;
	CRITICAL_SECTION gCS;
	 
	DWORD WINAPI Area(LPVOID pArg) {
	            int myNum = *((int *)pArg);
	            double h = 2.0 / NUM_RECT;
	            double partialSum = 0.0, x;  // local to each thread
	            int begin =  myNum	* (NUM_RECT / NUMTHREADS);
	            int end   = (myNum+1) * (NUM_RECT / NUMTHREADS);
	            if (nyNum == (NUMTHREADS-1)) end = NUM_RECT;
	            for ( int i = begin; i < end; ++i ){ //compute rectangles in range
	                  x = -1 + (i + 0.5f) * h;
	                  partialSum += sqrt(1.0f - x*x) * h; 
	            }
	            EnterCriticalSection(&gCS);
	            gPi += partialSum;  // add partial to global final answer
	            LeaveCriticalSection(&gCS);
	            return 0;
	}
	int main(int argc, char **argv) {
	            HANDLE threadHandles[NUMTHREADS];
	            int tNum[NUMTHREADS];
	            InitializeCriticalSection(&gCS);
	   	for ( int i = 0; i < NUMTHREADS; ++i ){
	      		tNum[i] = i;
	      		threadHandles[i] = CreateThread( NULL,  // Security attributes
	                                     0,   // Stack size
	                                 	   Area,   // Thread function
	                                 	  (LPVOID)&tNum[i],// Data for Area()
	                                 	   0,       // Thread start mode
	                                 	   NULL);   // Returned thread ID
	   }
	   WaitForMultipleObjects(NUMTHREADS, threadHandles, TRUE, INFINITE);
	   gPi * = 2.0;
	   DeleteCriticalSection(&gCS)
	   printf("Computed value of Pi:  %12.9f\n", gPi );
	}

