/*
  Original code provided by Dave Vaentine, Slippery Rock University.
  Edited by Libby Shoop, Macalester College.
*/
//
// Simulate many coin flips with rand_r() on multiple threads
// to determine how random the values are that are returned
// from each call.
//

#include <stdio.h>        // printf()
#include <stdlib.h>       // srand() and rand()
#include <time.h>         // time()

#include <omp.h>          // OpenMP functions and pragmas


//Standard chi sqaure test
double chiSq(int heads, int tails) {
    double sum = 0;                //chi square sum
    double tot = heads+tails;      //total flips
    double expected = 0.5 * tot;   //expected heads (or tails)
    
    sum = ((heads - expected)*(heads-expected)/expected) + \
        ((tails - expected)*(tails-expected)/expected);
    return sum;
}



int main() {
    int numFlips,             //loop control
        numHeads, numTails;   //counters
    
    /***  OMP ***/
    int nThreads;           // number of threads to use
    double ompStartTime, ompStopTime;  // holds wall clock time
    /***  OMP ***/


/***** Initialization *****/
    
    printf("Threaded Simulation of Coin Flip using rand_r()\n");
    /***  OMP ***/
    nThreads = 4;     // try increasing this if you have more cores
    
    //print our heading text
    printf("\n\n%15s%15s%15s%15s%15s%15s",
           "Trials","numHeads","numTails","total",
           "Chi Squared", "Time (sec)\n");
    
    
    //create seed using current time
    unsigned int seed = (unsigned) time(NULL);
    
    //create the pseudorandom number generator
    srand(seed);


// Try several trials of different numbers of flips doubling how many each round.
// 
// Use a unsigned int because we will try a great deal of flips for some trials.
    unsigned int trialFlips = 256;          // start with a smal number of flips
    unsigned int maxFlips = 1073741824;     // end with a very large number of flips
    
    // below we will double the number of trial flips and come back here
    // and run another trial, until we have reached > maxFlips.
    // This will be a total of 23 trials
    while (trialFlips <= maxFlips) {  
        
        numHeads = 0;               //reset counters
        numTails = 0;
        
        /***  OMP ***/
        ompStartTime = omp_get_wtime();   //get start time for this trial
    
    /***** Flip a coin trialFlips times, on each thread in parallel,
     *     with each thread getting its 1/4 share of the total flips.
     *****/

/***  OMP ***/
#pragma omp parallel for num_threads(nThreads) default(none) \
        private(numFlips, seed) \
        shared(trialFlips) \
        reduction(+:numHeads, numTails)
        for (numFlips=0; numFlips<trialFlips; numFlips++) {
            // rand() is not thread safe in linux
            // rand_r() is available in linux and thread safe,
            // to be run on separate threads concurrently.
            // On windows in visual studio, use rand(), which is thread safe.
            if (rand_r(&seed)%2 == 0) // if random number is even, call it heads
                numHeads++;       
            else
                numTails++;
        }
                
        /***  OMP ***/
        ompStopTime = omp_get_wtime();  //get time this trial finished
        
        // Finish this trial by printing out results

        printf("%15d%15d%15d%15d%15.6f%15.6f\n", trialFlips, numHeads, numTails,
               (numHeads+numTails), chiSq(numHeads, numTails),
               (double)(ompStopTime-ompStartTime));    /***  OMP ***/

        trialFlips *= 2;   // double the number of flips for the next trial
        srand(seed);      // start again with the same seed

    }
    
    /***** Finish Up *****/
    printf("\n\n\t<<< Normal Termination >>>\n\n");
    return 0;
}

