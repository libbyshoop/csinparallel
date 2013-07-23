#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>
using namespace std;

//compile commands:
//  standard   - icpc -O1 -std=c99 -DVERBOSE Magnitude_Momentum.cpp -o mm
//  vectorized - icpc -std=c99 -vec-report2 -DVERBOSE -DALIGNED Magnitude_Momentum.cpp -o mmVec

double clock_it(void){
  double duration = 0.0;
  struct timeval start;

  gettimeofday(&start, NULL);
  duration = (double)(start.tv_sec + start.tv_usec/1000000.0);
  return duration;
}

float ** readCols(int numWantedRows, int numCols, int numWantedCols, int* colNums, int headRows){
  fstream file;
  file.open("data2.txt");
  //skip through headRows
  char header[512];
  for (int i=0; i<headRows; i++)
    file.getline(header,512);
  //read in the wanted columns
  //initialize
  float ** cols = new float*[numWantedCols];
  for (int i=0; i<numWantedCols; i++)
    cols[i] = new float[numWantedRows];
  //read
  float temp;
  for (int i=0; i<numWantedRows; i++){
    for (int j=0; j<numCols; j++){
      file>>temp;
      for (int k=0; k<numWantedCols; k++){
        if (j==colNums[k]){
          cols[k][i]=temp;
        }
        else if (j<colNums[k])
          break;
      }
    }
  }
  return cols;
}

int main(){
  int numLines = 1800000;
  int headLen = 2;
  int numCols = 18;
  int numColsWanted = 3;
  int* colNums = new int[3];
  colNums[0]=13; colNums[1]=14; colNums[2]=15;

  float ** data;
  data=readCols(numLines,numCols,numColsWanted,colNums,headLen);

  double execTime;
  double startTime, endTime;

  //calculate a row-independent computation (and time it)
  startTime=clock_it(); //start timer

  // --- YOUR COMPUTATION HERE ---

  endTime=clock_it();
  execTime=endTime-startTime;

  //print results
  cout<<"Total time elapsed: "<<execTime<<" seconds"<<endl;
 
}
