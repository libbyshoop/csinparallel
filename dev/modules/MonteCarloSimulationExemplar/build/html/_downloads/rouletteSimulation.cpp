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
#include <omp.h>
#include <iomanip>
#include <time.h>
#include <string>
using namespace std;

const int MAX = 1<<28;	//limit to number of spins

//Function Prototypes
int getNumWins(int numSpins);
void showResults(int numSpins, int numWins);
int spinRed(int bet);
int randIntBetween(int low, int hi){
	return rand() % (hi-low+1) + low;
}

/**************************** MAIN **************
************************************************/
int main() {
	//Variable Dictionary
	int numSpins;		//#spins per trial
	int	numWins;		//#wins per trial
	clock_t startT, stopT; //wall clock elapsed time

/**************************** Initialization ****/
#pragma omp parallel
	{
		unsigned int seed = (unsigned) time(NULL);
		seed = (seed & 0xFFFFFFF0) | (omp_get_thread_num() + 1);
		srand(seed);
#pragma omp critical
		{
			cout<<"\n\n\nSeed "<<seed<<" for thread "<<omp_get_thread_num()
				<<" of "<<omp_get_num_threads()
				<<"  -first rand() is " << rand()<<endl;
		}
	}//parallel
	startT = clock();				//start our timer
	//srand((unsigned int)time(0));	//seed rand()
	cout<<"Simulation of an American Roulette Wheel\n" <<
		string(35,'*')<<endl;
	cout<<setw(12)<<"Num Spins" << setw(12) <<"Num Wins" << setw(12) <<"% wins"<<endl;
	numSpins = 1;					//we start with 1 spin (all or nothing)

/**************************** Do Simulations ***/
	while (numSpins < MAX) {
		numWins = getNumWins(numSpins);	//go spin wheel
		showResults(numSpins, numWins);				//show results

		numSpins += numSpins;	//double spins for next simulation
	}//while

/**************************** Finish Up ********/
	stopT= clock();		//stop our timer & show elapsed time
	cout<<"\nElapsed wall clock time: "<< (double)(stopT-startT)/CLOCKS_PER_SEC<<endl<<endl;

	cout<<"\n\n\n\t\t*** Normal Termination ***\n\n";
	return 0;
}//main


/*********************** getNumWins ************/ 
int getNumWins(int numSpins){
//always bet 'red' & count wins
	static int wins;//our counter
	int spin;		//loop cntrl var
	int myBet = 10; //amnt we bet per spin

	wins = 0;	//clear our counter

#pragma omp parallel for default(none) \
	shared(numSpins, myBet) \
	private(spin) \
	reduction(+:wins)

	for (spin=0; spin<numSpins; spin++){
		//spinRed returns +/- number (win/lose)
		if (spinRed(myBet) > 0) //a winner!
			wins++;
	}

	return wins;
}//getNumWins


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
	}while (n>0);
	return s;
}//prettyInt


/*********************** showResults ***********/
void showResults(int numSpins, int numWins){
//calc %wins & printout the 3 columns
	double percent = 100.0* (double)numWins/(double)numSpins;
	cout<<setw(12)<<prettyInt(numSpins) << setw(12) << 
		prettyInt(numWins) << setw(12) <<
		setprecision (4) << fixed << percent<< endl;
}//showResults


//spin the wheel, betting on RED
//Payout Rules:
//  0..17 you win (it was red)
// 18..35 you lose (it was black)
// 36..37 house wins (green) - you lose half
int spinRed(int bet) {
	int payout;
	int slot = randIntBetween(1,38);
	if (slot <= 18) //simplify odds: [0..17]==RED
		payout = bet;	//won
	else if (slot <= 36) //spin was 'black'-lose all
		payout = -bet;	//lost
	else //spin was green - lose half
		payout = -(bet/2); //half-back
	return payout;
}