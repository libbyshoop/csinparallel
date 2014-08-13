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
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iomanip>
#include <string>

#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

const int MAX_CARDS = 52;       //std deck of cards
const int MAX = 1<<20;          //max iterations is 1 meg
const int CARDS_IN_HAND = 4; //draw 4 cards at a time
const int NUM_SHUFFLES = 10; //num times to shuffle new deck

const int GRID_SIZE = 1;
const int BLOCK_SIZE = 128;

//Function prototypes...
__device__ int randIntBetween(int low, int hi, curandState *state);
__device__ void shuffleDeck(int deck[], int numCards, curandState *state);
__device__ void initDeck(int deck[], curandState *state);
__device__ int pickCard (int deck[], int& numCards, curandState *state);
__device__ void drawHand(int deck[], int hand[], curandState *state);
__device__ bool isFourSuits(int hand[]);
__device__ bool testOneHand(curandState *state);

__global__ void run_simulations(int n, curandState *state, unsigned int *result) {
	unsigned int tot = 0;
	int id = threadIdx.x;
	int thread2, halfPoint;
	int nTotalThreads = blockDim.x;
	__shared__ unsigned int sum[BLOCK_SIZE]; 
	int nTrials = n / nTotalThreads; 
	if(id == blockDim.x - 1) {
		nTrials += n % nTotalThreads;
	}
	curand_init(1235, id, 0, &state[id]);
	for(int i = 0; i < nTrials; i++) {
		if(testOneHand(&state[id])) { 
			tot++;
		}
	}
	
	sum[id] = tot;
	__syncthreads();
	while(nTotalThreads > 1) {
		halfPoint = (nTotalThreads >> 1);
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;
			sum[threadIdx.x] += sum[thread2];
		}
		__syncthreads();
		nTotalThreads = halfPoint;
	}
	
	if(id == 0) {
		*result = sum[0];
	}
}

/************************************************
******************************** M A I N *******/
int main() {
        int total;                      //num hands yielding 4 suits
        int numTests;           //#trials in each run
        int i;                          //lcv
        double percentage;      //% of hands with 4 suits
        clock_t startT, stopT;  //wallclock timer
	curandState *devStates;
	unsigned int *dev_result;
	unsigned int *result = (unsigned int*)malloc(sizeof(unsigned int));
	cudaMalloc((void **)&devStates, sizeof(curandState)*BLOCK_SIZE);
	cudaMalloc((void **)&dev_result, sizeof(unsigned int));
	//cudaMemset(dev_result, 0, sizeof(unsigned int));
	cudaDeviceSynchronize();
/************************* 1.0 Initialization **/
        startT=clock();         //start wallclock timer

        //print heading info...
        cout<<"\n\nDraw 4 suits with 4 cards...\n"<<string(50, '*')<<endl<<endl;
        cout<<setw(12)<<"numDraws" << setw(24) << "% draws with 4 suits" << endl;
 
/************************* 2.0 Simulation Loop **/
	numTests = 2048;
        while (numTests < MAX) {
		run_simulations<<<GRID_SIZE, BLOCK_SIZE>>>(numTests, devStates, dev_result);
	
		cudaDeviceSynchronize();		
                cudaMemcpy(result, dev_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		//calc % of 4-suit hands & report results...
                percentage = 100.0*((double) *result)/numTests;
                cout<<setw(12)<<numTests<<setw(14)<<setprecision(3 )<<fixed<<percentage<<endl;
                numTests += numTests;   //double #tests for next round
        }//while

/************************* 3.0 Finish up *******/
        stopT = clock();        //wallclock timer
	cudaFree(devStates);
	cudaFree(dev_result);
        cout<<"\nElapsed wallclock time: "<< (double)(stopT-startT)/CLOCKS_PER_SEC<<" seconds\n"<<endl;

        cout<<"\n\n\n\t\t*** Normal Termination ***\n\n";
        return 0;
}//main


/************************************************
***************************** randIntBetween ***/
__device__ int randIntBetween(int low, int hi, curandState *state){
//return random number in range [low..hi]
        return curand(state) % (hi-low+1) + low;
}


/************************************************
***************************** shuffleDeck ******/
__device__ void shuffleDeck(int deck[], int numCards, curandState *state) {
//simulate a shuffle like human would do it...
        int numIn20Percent = numCards/5;        //pick somewhere near middle
        int low = numIn20Percent*2;
        int hi = low + numIn20Percent-1;
        int mid = randIntBetween(low, hi, state);      //get split point in mid fifth of deck
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
                                if (curand(state)%2 > 0) {     //take card from low half
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
                                return;
                }//switch
        }//while
        delete [] temp; //garbage collect
}//shuffle


/************************************************
***************************** initDeck *********/
__device__ void initDeck(int deck[], curandState *state){
        int i;

        for (i=0; i<MAX_CARDS; i++)//load values
                deck[i]=i;

        for (i=0; i<NUM_SHUFFLES; i++){ //shuffle a bunch
                shuffleDeck(deck, MAX_CARDS, state);
        }
}//initDeck

/************************************************
***************************** pickCard *********/
__device__ int pickCard (int deck[], int& numCards, curandState *state){
//randomly pick 1 of numCards cards in deck
//remove card by copying 'tail' up one position in ary
        int loc = randIntBetween(0, numCards-1, state);
        int card = deck[loc];

        //remove card from deck by copying tail up 1 pos
        for (int i=loc+1; i<numCards; i++)
                deck[i-1]=deck[i];
        numCards--;

        return card;
}//pickCard


/************************************************
***************************** testOneHand *********/
__device__ bool testOneHand(curandState *state){
//Create a deck...sort it...pick 4 cards...test 4 suits
        int deck[MAX_CARDS];    //std deck
        int hand[CARDS_IN_HAND];        //card hand
        
        initDeck(deck, state); //create & shuffle a new deck
        
        drawHand(deck, hand, state);   //go pick cards from deck
        
        return isFourSuits(hand); //test if 4 suits
}//testOneHand

/************************************************
***************************** drawHand *********/
__device__ void drawHand(int deck[], int hand[], curandState *state){
//pick 5 cards w/out replacement from deck
        int i;
        int numCards = MAX_CARDS;
        int card;
        for (i=0; i<CARDS_IN_HAND; i++) {
                card = pickCard(deck, numCards, state);
                hand[i]=card;
        }
}//drawHand

//Find largest element in array of size 'n'
__device__ int findBiggest(int ary[], int n) {
        int big = ary[0]; //assume 1st is biggest
        for (int i=1; i<n; i++) //test remaining elem's
                if (ary[i]>big) 
                        big = ary[i];
        
        return big;
}

/************************************************
***************************** isFourSuits ********/
__device__ bool isFourSuits(int hand[]){
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

