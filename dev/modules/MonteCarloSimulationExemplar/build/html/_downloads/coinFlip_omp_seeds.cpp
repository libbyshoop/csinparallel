/*
  Original code provided by Dave Vaentine, Slippery Rock University.
  Edited by Libby Shoop, Macalester College.
*/
//
//  NOTE: this version is a variant of coinFlip_omp.cpp,
//        where we create a different seed for each thread.
//        

//
// Simulate many coin flips with rand_r() on multiple threads
// to determine how random the values are that are returned
// from each call.
//

#include <stdio.h>        // printf()
#include <stdlib.h>       // srand() and rand()
#include <time.h>        // time()

#include <omp.h>         // OpenMP functions and pragmas

/***  OMP ***/
const int nThreads = 4;  // number of threads to use
unsigned int seeds[nThreads];

void seedThreads() {
    int my_thread_id;
    unsigned int seed;
    #pragma omp parallel private (seed, my_thread_id)
    {
        my_thread_id = omp_get_thread_num();
        
        //create seed on thread using current time
        unsigned int seed = (unsigned) time(NULL);
        
        //munge the seed using our thread number so that each thread has its
        //own unique seed, therefore ensuring it will generate a different set of numbers
        seeds[my_thread_id] = (seed & 0xFFFFFFF0) | (my_thread_id + 1);
        
        printf("Thread %d has seed %u\n", my_thread_id, seeds[my_thread_id]);
    }
    
}
/***  OMP ***/

//Standard chi sqaure test
double chiSq(int heads, int tails) {
	double sum = 0;					//chi square sum
	double tot = heads+tails;		//total flips
	double expected = 0.5 * tot;	//expected heads (or tails)
	
	sum = ((heads - expected)*(heads-expected)/expected) + \
		((tails - expected)*(tails-expected)/expected);
	return sum;
}



int main() {
	int numFlips,			//loop control
		numHeads, numTails;	//counters
    
    /***  OMP ***/
    double ompStartTime, ompStopTime;   
    int tid;       // thread id when forking threads in for loop
    /***  OMP ***/
    
    unsigned int seed;   // seed each thread will use in for loop

/***** Initialization *****/

    /***  OMP ***/ 
    omp_set_num_threads(nThreads);  
    seedThreads();
    /***  OMP ***/ 
    
    
	printf("Threaded Simulation of Coin Flip using rand_r() and %d threads.\n", nThreads);
	
	//print our heading text
	printf("\n\n%15s%15s%15s%15s%15s%15s",
           "Trials","numHeads","numTails","total",
           "Chi Squared", "Time (sec)\n");
    
    

// Try several trials of different numbers of flips doubling how many each round.
// 
// Use a unsigned int because we will try a great deal of flips for some trials.
    unsigned int trialFlips = 256;          // start with a smal number of flips
    unsigned int maxFlips = 1073741824;     // this will be a total of 23 trials
    while (trialFlips <= maxFlips) {  // below we will double the number of trial flips
                                      // and come back here and run another trial,
                                      // until we have reached > maxFlips.
        
        numHeads = 0;               //reset counters
        numTails = 0;
        
        /***  OMP ***/
        ompStartTime = omp_get_wtime();   //get start time for this trial
    
    /***** Flip a coin trialFlips times, on each thread in parallel,
     *     with each thread getting its 1/4 share of the total flips.
     *****/
    /*
    #pragma omp parallel for default(none) \
        private(numFlips, seed) \
        shared(trialFlips, numHeads, numTails)
    */
    
/***  OMP ***/    
#pragma omp parallel num_threads(nThreads) default(none) \
        private(numFlips, tid, seed) \
        shared(trialFlips, seeds) \
        reduction(+:numHeads, numTails)
    {
        tid = omp_get_thread_num();   // my thread id
        seed = seeds[tid];            // it is much faster to keep a private copy of our seed
		srand(seed);	              //seed rand_r or rand
        
        #pragma omp for
        for (numFlips=0; numFlips<trialFlips; numFlips++) {
//          in Windows, can use rand()
//            if (rand()%2 == 0) // if random number is even, call it heads
            // linux: rand_r() is thread safe, to be run on separate threads concurrently
            if (rand_r(&seed)%2 == 0) // if random number is even, call it heads
                numHeads++;       
            else
                numTails++;
        }
        
    }
        /***  OMP ***/
        ompStopTime = omp_get_wtime();
        
        // Finish this trial by printing out results

        printf("%15d%15d%15d%15d%15.6f%15.6f\n", trialFlips, numHeads, numTails,
               (numHeads+numTails), chiSq(numHeads, numTails),
               (double)(ompStopTime-ompStartTime));    /***  OMP ***/

        trialFlips *= 2;   // double the number of flips for the next trial

        
    }

/***** Finish Up *****/
	printf("\n\n\t<<< Normal Termination >>>\n\n");
	return 0;
}

