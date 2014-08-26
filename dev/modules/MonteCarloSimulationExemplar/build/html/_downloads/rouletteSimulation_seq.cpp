/*
  Original code provided by Dave Vaentine, Slippery Rock University.
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
	
*/

#include <iostream>
#include <iomanip>
#include <time.h>
#include <string>
#include <stdlib.h>    //rand_r, rand
using namespace std;

const int MAX = 1<<30;	//limit to number of spins
                        // NOTE: change 30 to 29 or 28 for fewer trials
						//       so it will run faster


//Function Prototypes
int getNumWins(int numSpins, unsigned int seed);
void showResults(int numSpins, int numWins);
int spinRed(int bet, unsigned int* seed);
int rand_rIntBetween(int low, int hi, unsigned int* seed) {
	return rand_r(seed) % (hi-low+1) + low;
}
/* If use rand on Windows
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
	clock_t startT, stopT; //wall clock elapsed time

/**************************** Initialization ****/

/***** Initialization *****/
	startT = clock();     // start the timer
	
	//create seed for rand, rand_r using current time
    unsigned int seed = (unsigned) time(NULL);

	cout<<"Simulation of an American Roulette Wheel\n" <<
		string(35,'*')<<endl;
	cout<<setw(12)<<"Num Spins" << setw(12) <<"Num Wins" << setw(12) <<"% wins"<<endl;
	numSpins = 1;					//we start with 1 spin (all or nothing)

/**************************** Do Simulations ***/		
	while (numSpins < MAX) {
		srand(seed);	//seed rand_r() or rand()  for each trial
		numWins = getNumWins(numSpins, seed);	//go spin wheel
		showResults(numSpins, numWins);				//show results

		numSpins += numSpins;	//double spins for next simulation
	} //while

/**************************** Finish Up ********/
	stopT= clock();		//stop our timer & show elapsed time
	cout<<"\nElapsed wall clock time: "<< (double)(stopT-startT)/CLOCKS_PER_SEC<<endl<<endl;

	cout<<"\n\n\n\t\t*** Normal Termination ***\n\n";
	return 0;
} //main


/*********************** getNumWins ************/ 
int getNumWins(int numSpins, unsigned int seed) {
//always bet 'red' & count wins
	static int wins;//our counter
	int spin;		//loop cntrl var
	int myBet = 10; //amount we bet per spin

	wins = 0;	//clear our counter
	
	for (spin=0; spin<numSpins; spin++){
		//spinRed returns +/- number (win/lose)
		if (spinRed(myBet, &seed) > 0) //a winner!
			wins++;
	}
	
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
