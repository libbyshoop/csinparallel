/*
Libby-
I did this assignment after we had built a card game in class.  So many
of the helper functions (e.g. shuffle a deck) we had already developed.
If I was starting from scratch, I'd have taken some shortcuts, but this
way we demonstrated code reusability, etc.

As an aside, I was working these examples up at home where I have an 
AMD "APU" processor and this example just wouldn't seem to scale up
properly (1.25x improvement on 4-core).  It is the most "code intensive 
per thread" of my examples.  I did some digging & found that my 
AMD A-10-5800K shares 2 L1 caches between the 4 core (2x64k, 2-way 
set assoc, 64B lines).  I am guessing that I'm getting severe cache 
contention as core fight each other for control of the cache line 
holding the next instruction.  The code behaves as expected on my 
Intel laptop.  Interesting, eh?
*/




//Draw 4 cards from shuffled deck & test for Four Suits
// Probability is: 1 * (39/51) * (26/50) * (13/49) = 10.55%

#include <iostream>
#include <math.h>
#include <time.h>
#include <iomanip>
#include <string>

#include <mpi.h>

using namespace std;

#define FROM_MASTER 1   /* message sent from master */
#define FROM_WORKER 2   /* message sent from workers*/
#define MASTER 0        /* master has rank 0 */

const int MAX_CARDS = 52;       //std deck of cards
const int MAX = 1<<20;          //max iterations is 1 meg
const int CARDS_IN_HAND = 4; //draw 4 cards at a time
const int NUM_SHUFFLES = 10; //num times to shuffle new deck

MPI_Status status;      /* used to return data in recieving */

//Function prototypes...
int randIntBetween(int low, int hi);
void shuffleDeck(int deck[], int numCards);
void initDeck(int deck[]);
int pickCard (int deck[], int& numCards);
void drawHand(int deck[], int hand[]);
bool isFourSuits(int hand[]);
bool testOneHand();
void selectSort(int a[], int n);

int main(int argc, char *argv[]) {
	//initialize MPI environment
	int rank, size, mtype;
	 
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);


        int total;                      //num hands yielding 4 suits
        int partialSum;			//num hand yiedling 4 suits for each worker
	int numTests;           //#trials in each run
        int testsToRun;
	int i;                          //lcv
        double percentage;      //% of hands with 4 suits
        clock_t startT, stopT;  //wallclock timer

        startT=clock();         //start wallclock timer
        total = 0;                      //clear our counter
        numTests = 8;           //start with 8 trials

        //print heading info...
        cout<<"\n\nDraw 4 suits with 4 cards...\n"<<string(50, '*')<<endl<<endl;
        cout<<setw(12)<<"numDraws" << setw(24) << "% draws with 4 suits" << endl;
        
        while (numTests < MAX) {
                //reset counter
		partialSum == 0;
		
                testsToRun = numTests/size;
		if(rank == size - 1) testsToRun += numTest % size //assign remaining tests to last worker
	        for (i=0; i<testsToRun; i++) {
                        if (testOneHand())
                                partialSum++;
                }

		//The Master collects results and sums them
		if(rank == MASTER) {
			mtype = FROM_MASTER;
			total = partialSum;
			for(int source = 1; source < size; dest++) {
				MPI_Recv(&partialSum, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
				total += partialSum;
			}
		//Each work sends its partial sum to the Master
		} else {
			mtype = FROM_WORKER;
			MPI_Send(&partialSum, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

		}
                //calc % of 4-suit hands & report results...
                percentage = 100.0*( (double)total)/numTests;
                cout<<setw(12)<<numTests<<setw(14)<<setprecision(3 )<<fixed<<percentage<<endl;
                numTests += numTests;   //double #tests for next round
        }

        stopT = clock();        //wallclock timer
        cout<<"\nElapsed wallclock time: "<< (double)(stopT-startT)/CLOCKS_PER_SEC<<" seconds\n"<<endl;

        cout<<"\n\n\n\t\t*** Normal Termination ***\n\n";
        return 0;
}

//return random number in range [low..hi]
int randIntBetween(int low, int hi){
        return rand() % (hi-low+1) + low;
}

//simulate a shuffle like human would do it...
void shuffleDeck(int deck[], int numCards) {
        int numIn20Percent = numCards/5;        //pick somewhere near middle
        int low = numIn20Percent*2;
        int hi = low + numIn20Percent-1;
        int mid = randIntBetween(low, hi);      //get split point in mid fifth of deck
        int lowIndex = 0;       //start of LO half
        int hiIndex = mid;      //start of HIGH half
        int index = 0;          //loc in 'shuffled' deck

        enum STATE {MERGE2, FLUSH_HIGH, FLUSH_LOW, DONE}; //FSM to simulate fanning shuffle
        STATE myState = MERGE2;

        int *temp = new int[numCards];
        for (int i=0; i<numCards; i++)  //make copy of deck...shuffle back into orig deck
                temp[i]=deck[i];

        //FSM simulates a fanning-type shuffle
        while (myState != DONE) {
                switch (myState) {
                        case MERGE2:    //take one card from a half into new deck
                                if (rand()%2 > 0) {     //take card from low half
                                        deck[index]=temp[lowIndex];
                                        lowIndex++;
                                        if (lowIndex >= mid)    //last card in low half
                                                myState = FLUSH_HIGH;
                                }else {                         //take card from hi half
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
                        case FLUSH_HIGH:        //copy remaining cards in high half
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
                }
        }
        delete [] temp; 
}


void initDeck(int deck[]){
        int i;

        for (i=0; i<MAX_CARDS; i++)//load values
                deck[i]=i;

        for (i=0; i<NUM_SHUFFLES; i++){ //shuffle a bunch
                shuffleDeck(deck, MAX_CARDS);
        }
}//initDeck

int pickCard (int deck[], int& numCards){
//randomly pick 1 of numCards cards in deck
//remove card by copying 'tail' up one position in ary
        int loc = randIntBetween(0, numCards-1);
        int card = deck[loc];

        //remove card from deck by copying tail up 1 pos
        for (int i=loc+1; i<numCards; i++)
                deck[i-1]=deck[i];
        numCards--;

        return card;
}//pickCard

bool testOneHand(){
//Create a deck...sort it...pick 4 cards...test 4 suits
        int deck[MAX_CARDS];    //std deck
        int hand[CARDS_IN_HAND];        //card hand
        
        initDeck(deck); //create & shuffle a new deck
        
        drawHand(deck, hand);   //go pick cards from deck
        
        return isFourSuits(hand); //test if 4 suits
}//testOneHand

void drawHand(int deck[], int hand[]){
//pick 5 cards w/out replacement from deck
        int i;
        int numCards = MAX_CARDS;
        int card;
        for (i=0; i<CARDS_IN_HAND; i++) {
                card = pickCard(deck, numCards);
                hand[i]=card;
        }
}

//Find largest element in array of size 'n'
int findBiggest(int ary[], int n) {
        int big = ary[0]; //assume 1st is biggest
        for (int i=1; i<n; i++) //test remaining elem's
                if (ary[i]>big) 
                        big = ary[i];
        
        return big;
}

bool isFourSuits(int hand[]){
//convert cards to suit value (/ 13)


        int temp[4]={0};        //one for each suit

        //copy cards, converting to suit values
        for (int i=0; i<CARDS_IN_HAND; i++) {
                int suit = hand[i]/13;
                temp[suit]++;  //count the suits represented
        }
        
        //if largest #suits == 1 then all 4 suits counted in 4 cards
        return (findBiggest(temp, 4)==1);
}//isTwoPair

