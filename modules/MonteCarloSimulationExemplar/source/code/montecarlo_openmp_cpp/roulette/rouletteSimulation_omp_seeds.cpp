/*
  Original code provided by Dave Valentine, Slippery Rock University.
  Edited by Libby Shoop, Macalester College.
*/
/*
Simulate American Roulette wheel
	American wheel has 38 slots:
		-2 are 'green' (0 & 00)
			house ALWAYS wins on green
		-18 are red (1..18)
		-18 are black (1..18)

	Our user always bets "red"
	Odds should be:  18/38 or 47.37%
	
	This version maintains separate seeds for each thread
*/

#include <iostream>
#include <omp.h>
#include <iomanip>
#include <time.h>
#include <string>
#include <stdlib.h>    //rand_r, rand
#include <stdio.h>     //printf behaves a bit better with threads
using namespace std;

const int MAX = 1<<30;	//limit to number of spins
                        // NOTE: change 30 to 29 or 28 for fewer trials
						//       so it will run faster

/***  OMP ***/
const int nThreads = 4;  // number of threads to use
unsigned int seeds[nThreads];  // hold a seed per thread
/***  OMP ***/


//Function Prototypes
/***  OMP ***/
void seedThreads();

int getNumWins(int numSpins);
void showResults(int numSpins, int numWins);
int spinRed(int bet, unsigned int* seed);
int rand_rIntBetween(int low, int hi, unsigned int* seed){
	return rand_r(seed) % (hi-low+1) + low;
}
/* If use rand on Windows
 * Also must change where this is used below.
int randIntBetween(int low, int hi){
	return rand() % (hi-low+1) + low;
}
*/

/**************************** MAIN **************
************************************************/
int main() {
	//Variables
	int numSpins;		//#spins per trial
	int	numWins;		//#wins per trial
	// clock_t startT, stopT; //wall clock elapsed time

/**************************** Initialization ****/

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

	//srand((unsigned int)time(0));	//seed rand()
	cout<<"Simulation of an American Roulette Wheel\n" <<
		string(35,'*')<<endl;
	cout<<setw(12)<<"Num Spins" << setw(12) <<"Num Wins" << setw(12) <<"% wins"<<endl;
	numSpins = 1;					//we start with 1 spin (all or nothing)

/**************************** Do Simulations ***/
	/***  OMP ***/
    ompStartTime = omp_get_wtime();
		
	while (numSpins < MAX) {
		numWins = getNumWins(numSpins);	//go spin wheel
		showResults(numSpins, numWins);				//show results

		numSpins += numSpins;	//double spins for next simulation
	}//while

/**************************** Finish Up ********/
	/***  OMP ***/
    ompStopTime = omp_get_wtime();
	
	// Note: clock() doesn't work well for threaded code
	//stopT= clock();		//stop our timer & show elapsed time
	//cout<<"\nElapsed wall clock time: "<< (double)(stopT-startT)/CLOCKS_PER_SEC<<endl<<endl;
	cout<<"\nElapsed wall clock time: "<< (double)(ompStopTime - ompStartTime)<<endl<<endl;

	cout<<"\n\n\n\t\t*** Normal Termination ***\n\n";
	return 0;
}//main


/*********************** seedThreads ************/ 
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

/*********************** getNumWins ************/ 
int getNumWins(int numSpins) {
//always bet 'red' & count wins
	static int wins;//our counter
	int spin;		//loop cntrl var
	int myBet = 10; //amount we bet per spin

	wins = 0;	//clear our counter
	
	/***  OMP ***/
    int tid;       // thread id when forking threads in for loop
    unsigned int seed;   // seed each thread will use in for loop
	/*** OMP ***/

	/*
#pragma omp parallel for default(none) \
	shared(numSpins, myBet) \
	private(spin) \
	reduction(+:wins)
	*/
/***  OMP ***/    
#pragma omp parallel num_threads(nThreads) default(none) \
    shared(numSpins, myBet, seeds) \
	private(spin, seed, tid) \
	reduction(+:wins)
    {
        tid = omp_get_thread_num();   // my thread id
        seed = seeds[tid];            // it is much faster to keep a private copy of our seed
		srand(seed);	              //seed rand() or rand_r()
        
        #pragma omp for schedule(static) 
		for (spin=0; spin<numSpins; spin++){
			//spinRed returns +/- number (win/lose)
			if (spinRed(myBet, &seed) > 0) //a winner!
				wins++;
		}
	}
	////  end forked parallel threads
	
	return wins;
}  //getNumWins

//spin the wheel, betting on RED
//Payout Rules:
//  0..17 you win (it was red)
// 18..35 you lose (it was black)
// 36..37 house wins (green) - you lose half
int spinRed(int bet, unsigned int *seed) {
	int payout;
	int slot = rand_rIntBetween(1,38, seed);
	/* if Windows
	int slot = randIntBetween(1,38);
	 */
	if (slot <= 18) //simplify odds: [0..17]==RED
		payout = bet;	//won
	else if (slot <= 36) //spin was 'black'-lose all
		payout = -bet;	//lost
	else //spin was green - lose half
		payout = -(bet/2); //half-back
	return payout;
} // spinRed

/*********************** prettyInt *************/
string prettyInt(int n) {
//comma-delimited string made from int
	string s="";	//what we're making
	int digit;		//each digit of n
	int digitCnt=0; //count by 3's for comma insert

	do {
		digit = n % 10;		//get lsd
		n = n/10;			//and chop it from n
		//make digit into numeric char
		char c = (char) ( (int)'0' + digit);
		s.insert(0,1,c);	//insert char to string
		digitCnt++;			//count digits in string
		if ( (digitCnt%3 == 0) && (n>0) )
			s.insert(0,1,',');
	} while (n>0);
	return s;
} //prettyInt


/*********************** showResults ***********/
void showResults(int numSpins, int numWins){
//calc %wins & printout the 3 columns
	double percent = 100.0* (double)numWins/(double)numSpins;
	cout<<setw(12)<<prettyInt(numSpins) << setw(12) << 
		prettyInt(numWins) << setw(12) <<
		setprecision (4) << fixed << percent<< endl;
} //showResults
