/*
  Original code provided by Dave Vaentine, Slippery Rock University.
  Edited by Libby Shoop, Macalester College.
*/
//
// Simulate many coin flips with rand() to determine how
// random the values are that are returned from each call.
//

#include <stdio.h>        // printf()
#include <stdlib.h>       // srand() and rand()
#include <time.h>        // time()

//const int MAX = 1<<30; //1 gig

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
        clock_t startTime, stopTime; //wallclock timer

/***** Initialization *****/

    printf("Sequential Simulation of Coin Flip using rand()\n");
    
    //print our heading text
    printf("\n\n%15s%15s%15s%15s%15s%15s",
           "Trials","numHeads","numTails","total",
           "Chi Squared", "Time (sec)\n");

    //create seed using current time
    unsigned int seed = (unsigned) time(NULL);
    
    //create the pseudorandom number generator
    srand(seed);
    
// Try several trials of different numbers of flips, doubling how many each round.
// 
// Use a unsigned int because we will try a great deal of flips for some trials.
    unsigned int trialFlips = 256;       // start with a small number of flips
    unsigned int maxFlips = 1073741824;  // end with a very large number of flips
    
    // below we will double the number of trial flips and come back here
    // and run another trial, until we have reached > maxFlips.
    // This will be a total of 23 trials
    while (trialFlips <= maxFlips) {  
        // reset counters for each trial
        numHeads = 0;
        numTails = 0;
        startTime = clock();		//get start time for this trial
        
    /***** Flip a coin trialFlips times ****/
        for (numFlips=0; numFlips<trialFlips; numFlips++) {
            // if random number is even, call it heads
            // if (rand()%2 == 0)     // on Windows, use this
            if (rand_r(&seed)%2 == 0) // on linux, can use this
                numHeads++;
            else
                numTails++;
        }
        
        stopTime = clock();   // stop the clock
        
        /***** Show the results  for this trial  *****/
        printf("%15d%15d%15d%15d%15.6f%15.6f\n", trialFlips, numHeads, numTails,
               (numHeads+numTails), chiSq(numHeads, numTails),
               (double)(stopTime-startTime)/CLOCKS_PER_SEC);

        trialFlips *= 2;  // double the number of flips for the next trial
        srand(seed);      // start again with the same seed

    }

/***** Finish Up *****/
    printf("\n\n\t<<< Normal Termination >>>\n\n");
    return 0;
}
