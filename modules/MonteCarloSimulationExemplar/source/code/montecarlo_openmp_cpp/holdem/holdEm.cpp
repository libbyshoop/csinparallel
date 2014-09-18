//Compute odds of Texas Hold Em card hands
/*****************************************/
/** Cards are [0..51]                    **
***    so FACE is card % 13  [0..12]     **
***    and SUIT is card / 13 [0.. 4]     **
***                                      **
*** KING = face value of 12
*** ACE == face value of zero & can be   **
***   counted hi or low (special cases)  **
*** (yes, that does mean a '5' has a     **
*** face value of '6' but we never actu- **
*** ally use face for anything... we're  **
*** just counting hands)                 **
*******************************************
*** Verify results: 
http://en.wikipedia.org/wiki/Poker_probability
*********************************************/

#include <omp.h>
#include <stdio.h>
#include <time.h>

#define HAND_SIZE 7	//#cards in hand

//Function Prototypes
int findBiggest(int a[], int size);
bool is4Kind(int h[]);
bool is3Kind(int h[]);
bool isFullHouse(int h[]);
bool isTwoPair(int h[]);
bool isOnePair(int h[]);
bool isStraight(int h[]);
bool isRoyalFlush(int h[]);
bool isStraightFlush(int h[]);
bool isFlush(int h[]);

/*********************** MAIN **********
***************************************/
int main() {
	//Variable Dictionary
	int hand[HAND_SIZE];	//7 card hand (2 & 5)
	int h;			//loop cntrl var for outer loop (openMP req)
	int cnt, cntRoyalFlush, cntStraightFlush, cnt4Kind, 
		cntFullHouse, cntFlush, cntStraight, cnt3Kind, 
		cntTwoPair, cntOnePair, cntHiCard;
	int totHands=0;	//number of unique 7-card hands
	clock_t startT, stopT;	//wall clock elapsed time

/*********************** 1.0) Initialization ***/
	startT=clock();
	printf("Texax Hold'em 7 Card Poker Hands: computing the odds...\n");
	printf("=========================================================\n\n");

	//clear counters
	cnt= cntRoyalFlush= cntStraightFlush= cnt4Kind= cntFullHouse= cntFlush=\
		cntStraight= cnt3Kind= cntTwoPair= cntOnePair= cntHiCard=0;

/*********************** 2.0) Computation: brute force ***/

#pragma omp parallel for  default(none)  \
	shared(startT) \
	private(hand) \
	schedule(dynamic) \
	reduction(+:totHands, cntRoyalFlush, cntStraightFlush, cnt4Kind, cntFullHouse, \
	cntFlush,cntStraight, cnt3Kind, cntTwoPair, cntOnePair, cntHiCard) 
	
	//7 nested for-loops to generate 7-card hands
	for (h=0; h<46; h++) {
		hand[0]=h;		//openMP can't use subscripted index
		printf("Hand[0] = %d", hand[0]); //just for human
		printf("\tElapsed time is %f sec\n", (double)(clock()-startT)/CLOCKS_PER_SEC);

	  for (hand[1]=hand[0]+1; hand[1]<47; hand[1]++) 
	  for (hand[2]=hand[1]+1; hand[2]<48; hand[2]++) 
	  for (hand[3]=hand[2]+1; hand[3]<49; hand[3]++) 
	  for (hand[4]=hand[3]+1; hand[4]<50; hand[4]++)  
	  for (hand[5]=hand[4]+1; hand[5]<51; hand[5]++) 
	  for (hand[6]=hand[5]+1; hand[6]<52; hand[6]++) 
		{ 
			totHands++;		//count total hands
			//now test for each type of hand
			//NB: the hand is in ascending card order
			if (isRoyalFlush(hand))		cntRoyalFlush++;
			else if (isStraightFlush(hand)) cntStraightFlush++;
			else if (is4Kind(hand))		cnt4Kind++;
			else if (isFullHouse(hand)) cntFullHouse++;
			else if (isFlush(hand))		cntFlush++;
			else if (isStraight(hand))	cntStraight++;
			else if (is3Kind(hand))		cnt3Kind++;
			else if (isTwoPair(hand))	cntTwoPair++;
			else if (isOnePair(hand))	cntOnePair++;
			else cntHiCard++;
		}//for-hand[6]
			
	}//hand[0]
 
/*********************** 3.0) Show the results ***/
	stopT=clock();	//skip timing on final i/o
	printf("\n\nOur Texas Hold Em Results.....\n");
	printf("%15d Total hands\n\n", totHands);
	printf("%15d Royal Flush    (%7.4f%1c)\n", cntRoyalFlush, (double)cntRoyalFlush*100.0/(double)totHands,'%');
	printf("%15d Straight Flush (%7.4f%1c)\n", cntStraightFlush, (double)cntStraightFlush*100.0/(double)totHands,'%');
	printf("%15d Four of a Kind (%7.4f%1c)\n", cnt4Kind, (double)cnt4Kind*100.0/(double)totHands,'%');
	printf("%15d Full House     (%7.4f%1c)\n", cntFullHouse, (double)cntFullHouse*100.0/(double)totHands,'%');
	printf("%15d Flush          (%7.4f%1c)\n\n", cntFlush, (double)cntFlush*100.0/(double)totHands,'%');
	printf("%15d Straight       (%7.4f%1c)\n", cntStraight, (double)cntStraight*100.0/(double)totHands,'%');
	printf("%15d Three of Kind  (%7.4f%1c)\n", cnt3Kind, (double)cnt3Kind*100.0/(double)totHands,'%');
	printf("%15d Two Pair       (%7.4f%1c)\n", cntTwoPair, (double)cntTwoPair*100.0/(double)totHands,'%');
	printf("%15d One Pair       (%7.4f%1c)\n", cntOnePair, (double)cntOnePair*100.0/(double)totHands,'%');
	printf("%15d High Card      (%7.4f%1c)\n\n", cntHiCard, (double)cntHiCard*100.0/(double)totHands,'%');

	int testCnt;	//test to be sure we counted all the hands generated
	testCnt = cntRoyalFlush + cntStraightFlush+ cnt4Kind+ cntFullHouse+ cntFlush+ cntStraight+ cnt3Kind+ cntTwoPair+ cntOnePair+ cntHiCard;
	printf("%15d Total counted hands (%d unaccounted)\n", testCnt, (totHands-testCnt));

	printf("\nElapsed time is %f sec\n", (double)(stopT-startT)/CLOCKS_PER_SEC);
	printf("\n\n***Normal Termination ***\n\n");

	return 0;
}//Main



int findBiggest(int a[], int size) {
//Return biggest valu in a[size]
	int big = a[0]; //assume first elem is big
	int i;
	for (i=1; i<size; i++) //loop from [1]
		if (a[i]>big)
			big=a[i];
	return big;
}//findBiggest

int findBigIndex(int a[], int size) {
//Return index of biggest elem in a[size]
	int bigIndex = 0;	//assume 1st elem is big
	
	int i;
	for (i=1; i<size; i++) 
		if (a[i]>a[bigIndex])
			bigIndex=i;
	return bigIndex;
}

bool is4Kind(int h[]) {
	//Var dictionary
	int face[13]={0}; //13 possible face valu
	int i, big;
	for (i=0; i<HAND_SIZE; i++){	//count number of each face valu
		int myFace = h[i]%13; //get face valu
		face[myFace]++;
	}
	big = findBiggest(face,13);	//now find face with most values
	
	return (big>=4);	//4 or more is 4-of-a-kind
}
bool is3Kind(int h[]) {
//see is4Kind
	int face[13]={0};
	int i, big;
	for (i=0; i<HAND_SIZE; i++){
		int myFace = h[i]%13; //get face valu
		face[myFace]++;
	}
	big = findBiggest(face,13);
	
	return (big>=3);
}

bool isFullHouse(int h[]) {
	//Var dictionary
	bool flag = false;	
	int face[13]={0};	//13 possible face valus
	int i, bigIndex;
	
	for (i=0; i<HAND_SIZE; i++){	//count each face valu in hands
		int myFace = h[i]%13; //get face valu
		face[myFace]++;
	}
	bigIndex = findBigIndex(face,13);
	if (face[bigIndex] >= 3) {//potential FH
		face[bigIndex]=0; //clear the 3
		bigIndex=findBigIndex(face,13); //find next biggest one
		if (face[bigIndex]>=2) //we have a pair (at least)
			flag=true;
	}//found at least 3
	
	return flag;
}//isFullHouse

bool isTwoPair(int h[]) {
	//Var Dictionary
	bool flag = false;	//assume NO 2-pair
	int face[13]={0};	//count of each face valu
	int i, bigIndex;
	
	for (i=0; i<HAND_SIZE; i++){ //count each face valu
		int myFace = h[i]%13; //get face valu
		face[myFace]++;
	}
	//get index of biggest face valu
	bigIndex = findBigIndex(face,13);
	
	if (face[bigIndex] >= 2) {//have at least 1 pair
		face[bigIndex]=0; //clear the 'pair'
		bigIndex=findBigIndex(face,13); //find next biggest one
		if (face[bigIndex]>=2) //have another 'pair'
			flag=true;
	}//found at least 3
	
	return flag;
}//isTwoPair


bool isOnePair(int h[]) {
	int face[13]={0};
	int i, big;
	for (i=0; i<HAND_SIZE; i++){
		int myFace = h[i]%13; //get face valu
		face[myFace]++;
	}
	big = findBiggest(face,13);
	
	return (big >= 2);
}//isOnePair


bool isAllThere(int h[], int lo, int hi){
//test if each elem in h[lo..hi] is > 0
//meaning we have at least 1 card in each slot
	bool flag=true;
	int i=lo;
	
	while (flag && (i<=hi)){
		if (h[i]>0) //is there
			i++;
		else
			flag=false;
	}
	return flag;
}

bool isStraight(int h[]) {
//test straight: linear sequence
	int face[13]={0};
	int i;
	
	for (i=0; i<HAND_SIZE; i++){ //count face values
		int myFace = h[i]%13; //get face valu
		face[myFace]++;
	}

	if (isAllThere(face,0,4)) return true;
	if (isAllThere(face,1,5)) return true;
	if (isAllThere(face,2,6)) return true;
	if (isAllThere(face,3,7)) return true;
	if (isAllThere(face,4,8)) return true;
	if (isAllThere(face,5,9)) return true;
	if (isAllThere(face,6,10)) return true;
	if (isAllThere(face,7,11)) return true;
	if (isAllThere(face,8,12)) return true;
	//special case of ACE high...just move ace to #8
	//...will make A-10-J-Q-K
	face[8] = face[0];
	if (isAllThere(face,8,12)) return true;

	return false;
}//isStraight

	

//isInHand(h, 7, (lastCard-12)
bool isInHand(int h[], int size, int valu) {
	bool flag = false;
	int i=0;
	while ( (i<size) && (!flag) )
		if (h[i]==valu)
			flag = true;
		else
			i++;
	return flag;
}


bool isRoyalFlush(int h[]){
//special case testing: brute force
// royal in suit 0 = face(0,9,10,11,12)
// royal in suit 1 = face(13,22,23,24,25), etc
	bool flag = false;
	if (isInHand(h, 7, 0) && isInHand(h, 7, 12) && isInHand(h, 7, 11) && isInHand(h, 7, 10) && isInHand(h, 7, 9) )
		flag=true;
	else if (isInHand(h, 7, 13) && isInHand(h, 7, 25) && isInHand(h, 7, 24) && isInHand(h, 7, 23) && isInHand(h, 7, 22) )
		flag = true;
	else if (isInHand(h, 7, 26) && isInHand(h, 7, 38) && isInHand(h, 7, 37) && isInHand(h, 7, 36) && isInHand(h, 7, 35) )
		flag = true;
	else if (isInHand(h, 7, 39) && isInHand(h, 7, 51) && isInHand(h, 7, 50) && isInHand(h, 7, 49) && isInHand(h, 7, 48) )
		flag = true;

	return flag;
}

bool isStraightFlush(int h[]){
	//use FSM to test
	//the hand is in ascending card order (built in for-loops)
	enum STATE  {INIT, COUNTING, TALLY, DONE};
	STATE fsm = INIT;
	bool flag = false;  //neither kind
	int index, cnt,lastCard;
	while (fsm != DONE) {
		switch (fsm) {
		case INIT:
			index = 1;
			cnt = 1;
			lastCard=h[0];
			fsm = COUNTING;
			break;
		case COUNTING: //count linear sequence in same suit
			if (index >= HAND_SIZE) //ran out of cards
				fsm = TALLY;
			else if ( (lastCard/13) != (h[index]/13) )//suit change
				fsm = TALLY;
			else if (h[index]==(lastCard+1)) {//still in straight
				cnt++; lastCard=h[index]; index++;
			}else fsm = TALLY; //not in straight
			break;
		case TALLY: //how big a linear sequence?
			if (cnt >= 5) {
				flag = true;  //got 5 in row in same suit
				fsm = DONE;
			} else if ( (lastCard%13 == 12) && (cnt == 4) && isInHand(h, 7, (lastCard-12) )) {//special cases
				flag = true; //ace high will make a sequence in this suit
				fsm = DONE;
			} else if (index >= HAND_SIZE) {//out of cards
				flag = false;
				fsm = DONE;
			} else { //reset & start counting again
				cnt=1; lastCard=h[index]; index++;
				fsm = COUNTING;
			}
			break;
		default:
			printf("\nIllegal fsm state in isStraightFlush\n");
			return false;
		}//switch
	}//while
	/*if (flag){
		for(cnt=0; cnt<7; cnt++) printf("%3d", h[cnt]);
		printf(" is a straight flush\n");
	}*/
	return flag;
}


bool isFlush(int h[]){
//do we have 5 of same suit?
	int suits[4]={0};
	int i, big;
	
	for (i=0; i<HAND_SIZE; i++) //count suits
		suits[h[i]/13]++;
	
	big = findBiggest(suits, 4);
	return (big>=5); //five or more in same suit
}

