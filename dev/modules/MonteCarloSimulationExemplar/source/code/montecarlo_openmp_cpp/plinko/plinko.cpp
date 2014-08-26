 //Simulate PLINKO from Price-is-right...
//cf.  http://www.mathdemos.org/mathdemos/plinko/
//		http://www.mathdemos.org/mathdemos/plinko/bigboardplinko.html
/*********************************************************/
/** Find the diagram given at this URL:                  **
*** The even rows have 9 squares [A..I] and the odd rows **
*** have 8 squares [J..Q].                               **
*** Moving from even row to odd row, the two outer-most  **
*** squares are forced "in" (to J or Q) by the bumper and**
*** all the interior squares have 50/50 chance of bounce **
*** left or right.                                       **
*** Moving from odd to even rows gives every square      **
*** a 50/50 chance of bouncing left/right.               ** 
***********************************************************
*** Our solution uses the int's [0..16] to hold one set  **
*** of even/odd rows: 0->A, 1->B, ... 16->Q.  Picture as **
***      0   1   2   3   4   5   6   7   8               **
***        9   10  11  12  13  14  15  16                **
***                                                      **
*** So a disk in 2 may bounce to 10 or 11 (2+8 or 2+9)   **
*** and a disk in 6 may go to 14 or 15 (6+8 or 6+9)      **
*** But a disk in 0 or 8 must bounce to only 9 or 16     **
***                                                      **
*** Going the OTHER way (odd to even row) see:           **
***        9   10  11  12  13  14  15  16                **
***      0   1   2   3   4   5   6   7   8               **
***                                                      **
*** A disk in 10 may go to 1 or 2 (10-9 or 10-8) and     **
*** a disk in 16 may go to 7 or 8 (16-9 or 16-8)         **
**********************************************************/
#include <iostream>
#include <omp.h>
#include <time.h>
#include <iomanip>
using namespace std;

const int MAX = 1<<20;		//max disks per test
const int NUM_TALLY=9;		//# of ending bins
const int BIG_MONEY=10;		//scaled from 10k

//Function prototypes
char evenToOddRow (char pos);
char oddToEvenRow (char pos);
double dropOneDisk(char startPos);
void dropDisks(char pos, int numDisks, int numBigMoney[]);
void showResults(int numDisks, int numBigMoney[]);

/****************************** MAIN ************
************************************************/
int main() {
	int numDisks;			//number of disks to drop in this test
	int numBigMoney[NUM_TALLY]={0};	//hits on BIG MONEY per starting col
	char pos;				//starting colum ['A'..'I']
	clock_t startT, stopT;	//wallclock time

/****************** Initialization **************/	
	numDisks = 5;		//begin with 5 disks (like the show)
	
	//seed each thread's rand()
#pragma omp parallel
	{
		unsigned int seed = (unsigned) time(NULL);
		seed = (seed & 0xFFFFFFF0) | (omp_get_thread_num() + 1);
		srand(seed);
#pragma omp critical
		{
			cout<<"Seed "<<seed<<" for thread "<<omp_get_thread_num()
				<<" of "<<omp_get_num_threads()<<endl;
		}
	}//parallel

	//print header...
	cout<<"\n\n\nSimulate PLINKO from Price-is-right... \n"<<
		"\tcf.  http://www.mathdemos.org/mathdemos/plinko/\n"<<
		"***********************************************\n\n";
	cout<<"\nTesting "<<scientific<<setprecision(2)<<(double)numDisks<<" disks per slot\n\n";
/****************** Run a LOT of tests, with increasing # of disks **/
	startT = clock(); //start wallclock timer

	while (numDisks < MAX) {
		//drop numDisks thru each starting pos [A..I]
		cout<<"\n\nDropping " << numDisks << " disks---\n";
		for(pos='A'; pos<='I'; pos++) {
			dropDisks(pos, numDisks, numBigMoney);
		}//for-pos

		//show totals for this run...
		showResults(numDisks, numBigMoney);
		
		//increase #disks for next run
		numDisks+=numDisks;
	}//while
	

/****************** Finish Up ******************/
	stopT=clock();
	cout<<"\n\nElapsed time: " << (double)(stopT-startT)/CLOCKS_PER_SEC<<" sec\n";

	cout<<"\n\n\t\t*** Normal Termination ***\n\n";
	return 0;
}//Main


/** move disk from even row to odd row.... **/
char evenToOddRow (char pos){
//brute force switch statement
	char newPos;
	switch(pos) {
		case 'A':	//bumper forces to J
			newPos='J'; break;
		case 'B':	if (rand()%2==0)
						newPos='J';
					else
						newPos='K';
			break;
		case 'C':	if (rand()%2==0)
						newPos='K';
					else
						newPos='L';
			break;
		case 'D':	if (rand()%2==0)
						newPos = 'L';
					else
						newPos = 'M';
			break;
		case 'E':	if (rand()%2==0)
						newPos='M';
					else newPos='N';
					break;
		case 'F':	if (rand()%2==0)
						newPos='N';
					else newPos='O';
					break;
		case 'G':	if (rand()%2==0)
						newPos='O';
					else newPos='P';
					break;
		case 'H':	if (rand()%2==0)
						newPos='P';
					else newPos='Q';
					break;
		case 'I':	//bumper forces to Q
			newPos='Q';
			break;
		default:
			cerr<<"\nBad value in evenToOddRow: " << pos <<endl;
			newPos='M';
	}//switch
	return newPos;
}//evenToOddRow


/** Move disk from odd row to even row ....**/
char oddToEvenRow (char pos){
//smarter transition than evenToOddRow
//move to letter at -9 or -8 from cur pos
	char newPos;
	newPos =(char) (((int)pos - 9) + (rand()%2));
	return newPos;
}

/** move a disk from starting row to ending row & **
*** return the dollars won                        */
double dropOneDisk(char pos){
	int row;		//count 12 rows
	double valu;	//return value (scaled in 1000's)
	//bounce disk thru 12 rows of Plinko Board
	for (row=0; row<12; row++) {
		if (row%2==0) //even row to odd row
			pos = evenToOddRow(pos);
		else
			pos = oddToEvenRow(pos);
	}
	//calculate winnings based on ending pos
	//We had integer overflow problems, so converted
	//	to scaled double (returns valu in thousands)
	switch (pos) {
		case 'A':
		case 'I':	valu = 0.1;
			break;
		case 'B':
		case 'H':	valu = 0.5;
			break;
		case 'C':case 'G':	valu = 1.0;
			break;
		case 'D': case'F':	valu = 0;
			break;
		case 'E':		valu = BIG_MONEY;
			break;
		default:
			cerr<<"\nBad value ending dropOneDisk: "<<pos<<endl;
			valu=-1;
	}//switch
	return valu;

}//dropOneDisk

/** Drop numDisk disks in pos, tracking #hits in BigMoney **/
void dropDisks(char pos, int numDisks, int numBigMoney[]){
	static double totalWon=0.0;		//clear the counters
	int numBigMoneyHits=0;
	int disk;
	
	//convert char starting pos to int index for array
	int index = (int)pos - (int)'A';

#pragma omp parallel for default(none) \
	shared (numDisks, pos, cout) \
	private (disk) \
	reduction(+:totalWon, numBigMoneyHits)

	//The workhorse loop
	for (disk=0; disk<numDisks; disk++) {
		double valu = dropOneDisk(pos); //how much did we win?
		totalWon+=valu;			//keep running total for pos
		if (valu==BIG_MONEY)	//was it the BIG MONEY?
			numBigMoneyHits++;
	}//for-disk

	numBigMoney[index]=numBigMoneyHits;	//tally bigMoney hits for pos
	
	//show results
	cout<<"Dropping in "<<pos<<" won $" <<
		scientific <<setprecision(2)<< totalWon*1000<<endl;
}//dropDisks

/** Show the %results to human **/
void showResults(int numDisks, int numBigMoney[]){
	int lcv;	//loop control var

	cout<<"\nNum hits in 'Big Money' bin...\n";
	cout<<setw(4)<<"pos"<<setw(12)<<"#bigMoney"<<setw(12)<<"%Hits"<<endl;
	for (lcv=0; lcv<NUM_TALLY; lcv++){
		cout<<setw(4)<<(char)(lcv+(int)'A')<<setw(12)<<numBigMoney[lcv]<<
			setw(12)<<setprecision(3)<<fixed << 1e2*numBigMoney[lcv]/(double)numDisks<<"%\n";
		numBigMoney[lcv]=0;	//clear tally for next run
	}
	cout<<endl<<"********************************************\n";
}//showResults


