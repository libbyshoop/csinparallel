/*
  Original code provided by Dave Valentine, Slippery Rock University.
  Edited by Libby Shoop, Macalester College.
*/

//Draw 4 cards from shuffled deck & test for Four Suits
// Probability is: 1 * (39/51) * (26/50) * (13/49) = 10.55%

#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <iomanip>
#include <string>
using namespace std;

const int MAX_CARDS = 52;	//std deck of cards
const int MAX = 1<<20;		//max iterations is 1 meg
const int CARDS_IN_HAND = 4; //draw 4 cards at a time
const int NUM_SHUFFLES = 10; //num times to shuffle new deck

//Function prototypes...
bool isFourSuits(int hand[]);
void selectSort(int a[], int n);

/*  Windows versions using rand()
int randIntBetween(int low, int hi);
void shuffleDeck(int deck[], int numCards);
void initDeck(int deck[]);
int pickCard (int deck[], int& numCards);
void drawHand(int deck[], int hand[]);
bool testOneHand();
*/

//linux versions for rand_r with a seed
int rand_rIntBetween(int low, int hi, unsigned int *seed_ptr);
void shuffleDeck(int deck[], int numCards, unsigned int *seed_ptr);
void initDeck(int deck[], unsigned int *seed_ptr);
int pickCard (int deck[], int& numCards, unsigned int *seed_ptr);
void drawHand(int deck[], int hand[], unsigned int *seed_ptr);
bool testOneHand(unsigned int *seed_ptr);

/***  OMP ***/
const int nThreads = 4;   //number of threads to use

/************************************************
******************************** M A I N *******/
int main() {
	int total;			//num hands yielding 4 suits
	int numTests;		//#trials in each run
	int i;				//lcv
	double percentage;	//% of hands with 4 suits
	//clock_t startT, stopT;	//wallclock timer
    /***  OMP ***/
	double ompStartTime, ompStopTime;  // holds wall clock time

/************************* 1.0 Initialization **/
	//startT=clock();		//start wallclock timer
	/***  OMP ***/
	ompStartTime = omp_get_wtime();   //get start time for this trial
	
	total = 0;			//clear our counter
	numTests = 8;		//start with 8 trials

	//  seed the random number generator for all trials
	unsigned int seed = (unsigned) time(NULL);
	srand(seed);

	//print heading info...
	cout<<"\n\nDraw 4 suits with 4 cards...\n"<<string(50, '*')<<endl<<endl;
	cout<<setw(12)<<"numDraws" << setw(24) << "% draws with 4 suits" << endl;
	

/************************* 2.0 Simulation Loop **/
	while (numTests < MAX) {
		total = 0;	//reset counter
		
#pragma omp parallel for num_threads(nThreads) default(none) \
		private (i, seed) \
		shared (numTests) \
		reduction (+:total)	\
		schedule(dynamic)
		for (i=0; i<numTests; i++) { //make new deck - pick hand - test for 4 suits
			if (testOneHand(&seed))		//returns TRUE iff 4-suits hand
				total ++;		//tally hands with 4-suits
		}
		//calc % of 4-suit hands & report results...
		percentage = 100.0*( (double)total)/numTests;
		cout<<setw(12)<<numTests<<setw(14)<<setprecision(3 )<<fixed<<percentage<<endl;
		numTests += numTests;	//double #tests for next round
	} //while

/************************* 3.0 Finish up *******/
	//stopT = clock();	//wallclock timer
	/***  OMP ***/
	ompStopTime = omp_get_wtime();  //get time this trial finished
	cout<<"\nElapsed wallclock time: "<< (double)(ompStopTime-ompStartTime)<<" seconds\n"<<endl;

	cout<<"\n\t\t*** Normal Termination ***\n\n";
	return 0;
} //main


/************************************************
***************************** randIntBetween ***/
/* Windows
int randIntBetween(int low, int hi){
//return random number in range [low..hi]
	return rand() % (hi-low+1) + low;
}
*/

/************************************************
***************************** rand_rIntBetween ***/
/* linux */
int rand_rIntBetween(int low, int hi, unsigned int *seed){
//return random number in range [low..hi]
	return rand_r(seed) % (hi-low+1) + low;
}


/************************************************
***************************** shuffleDeck ******/
void shuffleDeck(int deck[], int numCards, unsigned int *seed_ptr) {
//simulate a shuffle like human would do it...
	int numIn20Percent = numCards/5;	//pick somewhere near middle
	int low = numIn20Percent*2;
	int hi = low + numIn20Percent-1;
	int mid = rand_rIntBetween(low, hi, seed_ptr);	//get split point in mid fifth of deck
	int lowIndex = 0;	//start of LO half
	int hiIndex = mid;	//start of HIGH half
	int index = 0;		//loc in 'shuffled' deck

	enum STATE {MERGE2, FLUSH_HIGH, FLUSH_LOW, DONE}; //FSM to simulate fanning shuffle
	STATE myState = MERGE2;

	int *temp = new int[numCards];
	for (int i=0; i<numCards; i++)	//make copy of deck...shuffle back into orig deck
		temp[i]=deck[i];

	//FSM simulates a fanning-type shuffle
	while (myState != DONE) {
		switch (myState) {
			case MERGE2:	//take one card from a half into new deck
				// swutch these next two if on Windows
//				if (rand()%2 > 0) {	//take card from low half
				if (rand_r(seed_ptr)%2 > 0) {	//take card from low half
					deck[index]=temp[lowIndex];
					lowIndex++;
					if (lowIndex >= mid)	//last card in low half
						myState = FLUSH_HIGH;
				}else {				//take card from hi half
					deck[index]=temp[hiIndex];
					hiIndex++;
					if (hiIndex >= numCards) //last card in high half
						myState = FLUSH_LOW;
				}
				index++;
				break;
			case FLUSH_LOW:
				while (index<numCards) {//copy remaining cards in low half
					deck[index] = temp[lowIndex];
					lowIndex++;
					index++;
				}
				myState = DONE;
				break;
			case FLUSH_HIGH:	//copy remaining cards in high half
				while (index<numCards) {
					deck[index] = temp[hiIndex];
					hiIndex++;
					index++;
				}
				myState = DONE;
				break;
			default:
				cerr<<"\nBad state in FSM\n";
				return;
		}//switch
	}//while
	delete [] temp;	//garbage collect
}//shuffle


/************************************************
***************************** initDeck *********/
void initDeck(int deck[], unsigned int *seed_ptr){
	int i;

	for (i=0; i<MAX_CARDS; i++)//load values
		deck[i]=i;

	for (i=0; i<NUM_SHUFFLES; i++){ //shuffle a bunch
		shuffleDeck(deck, MAX_CARDS, seed_ptr);
	}
}//initDeck

/************************************************
***************************** pickCard *********/
int pickCard (int deck[], int& numCards, unsigned int *seed_ptr){
//randomly pick 1 of numCards cards in deck
//remove card by copying 'tail' up one position in ary
	int loc = rand_rIntBetween(0, numCards-1, seed_ptr);
	int card = deck[loc];

	//remove card from deck by copying tail up 1 pos
	for (int i=loc+1; i<numCards; i++)
		deck[i-1]=deck[i];
	numCards--;

	return card;
}//pickCard


/************************************************
***************************** testOneHand *********/
bool testOneHand(unsigned int *seed_ptr){
//Create a deck...sort it...pick 4 cards...test 4 suits
	int deck[MAX_CARDS];	//std deck
	int hand[CARDS_IN_HAND];	//card hand
	
	initDeck(deck, seed_ptr);	//create & shuffle a new deck
	
	drawHand(deck, hand, seed_ptr);	//go pick cards from deck
	
	return isFourSuits(hand); //test if 4 suits
}//testOneHand

/************************************************
***************************** drawHand *********/
void drawHand(int deck[], int hand[], unsigned int *seed_ptr){
//pick 5 cards w/out replacement from deck
	int i;
	int numCards = MAX_CARDS;
	int card;
	for (i=0; i<CARDS_IN_HAND; i++) {
		card = pickCard(deck, numCards, seed_ptr);
		hand[i]=card;
	}
}//drawHand

//Find largest element in array of size 'n'
int findBiggest(int ary[], int n) {
	int big = ary[0]; //assume 1st is biggest
	for (int i=1; i<n; i++) //test remaining elem's
		if (ary[i]>big) 
			big = ary[i];
	
	return big;
}

/************************************************
***************************** isFourSuits ********/
bool isFourSuits(int hand[]){
//convert cards to suit value (/ 13)


	int temp[4]={0};	//one for each suit

	//copy cards, converting to suit values
	for (int i=0; i<CARDS_IN_HAND; i++) {
		int suit = hand[i]/13;
		temp[suit]++;  //count the suits represented
	}
	
	//if largest #suits == 1 then all 4 suits counted in 4 cards
	return (findBiggest(temp, 4)==1);
}//isTwoPair

