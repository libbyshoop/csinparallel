/* PDC project S13 by Luke Bonde and Allison Brumfield   4/17/2013
   Simulates an epidemic or disease spreading through a population */


#include <iostream>
using namespace std;
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>

const int width = 10000;    //The width of the environment
const int height = 10000;   //The height of the environemnt
const int head = 0;

// The state of health a person can be in
enum State {Susceptible = 0, Infected = 1, Recovered = 2};

//A class that contains the pertinate information about the specific disease
class Infection {
public:
  int duration;          // 1 unit of time = 6 hours
  float contagiousness;  // Percent chance of transmission
  float radius;            // The radius in which transimission is possible
  Infection(int d, float c, float r) {
    duration = d;
    contagiousness = c;
    radius = r;}
} ;

//A class that represents a single person that can move around randomly in the world and potentially get sick
class Person {
public:
  int x,y;
  State state;
  int infectedPeriod; // Units of time until recovered.
  Person() { 
    x = rand() % width; 
    y = rand() % height; 
    state = Susceptible;}
    void updateState(State s) {
    state = s;}
  void infectWith(Infection i){
    updateState(Infected);
    infectedPeriod = i.duration;
  }
  bool isInfected(){
    return (state==Infected);
  }
  bool isSusceptible(){
    return (state==Susceptible);
  }
  void move() { 
    x = (x + (rand() % 5) - 2 + width) % width ; // Add width to ensure non-negativity...
    y = (y + (rand() % 5) - 2 + height) % height;
  }
  void timeStep() { 
    move();
    if (infectedPeriod > 0) --infectedPeriod;
    else if (infectedPeriod == 0 and isInfected()) updateState(Recovered);
  }
};

float dist(Person & p1, int x, int y) {
  float dist = pow((p1.x-x),2)+pow((p1.y-y),2);
  return sqrt(dist);
}
  

int main(int argc, char** argv) {
  MPI::Init(argc,argv);
  int my_rank = MPI::COMM_WORLD.Get_rank();
  int nprocs = MPI::COMM_WORLD.Get_size();

  srand(time(NULL)*my_rank+my_rank);

  // Set up parameters. This information is needed by each node.
  int numPersons = 20000;
  int initialInfected = 100;
  int numIterations = 200;
  Infection Influenza(120,.5,45);
  int maxInfected = 0;
  int personPortion = (int) numPersons/nprocs;
  int infectedPortion = (int) initialInfected/nprocs;
  Person * MN;
  int * x_pos, * y_pos;
  int * all_x, * all_y;
  int n; // Counter: counts number of infected persons
  
  // Head node takes remainder if nprocs does not divide persons
  if (my_rank==head) {
    int personsLeft = numPersons - personPortion*nprocs; 
    personPortion += personsLeft;
    int infectedLeft = initialInfected - infectedPortion*nprocs;
    infectedPortion += infectedLeft;
  }

  // Initialize Person arrays
  MN = new Person[personPortion];
  for (int i = 0; i < infectedPortion; ++i)
      MN[i].infectedWith(Influenza);
  
  ////////////////// BEGIN SIMULATION ///////////////////

  for (int i = 0; i < numIterations; ++i) {
    
    // Calculate the max number of infected people on an individula process. Allreduce is a collective communication that performs a reduce, then a scatter.
    MPI::COMM_WORLD.Allreduce(&infectedPortion,&maxInfected,1,MPI::INT,MPI::MAX);
    // Set up an array to hold the positions on infected people on this node.
    x_pos = new int[maxInfected];
    y_pos = new int[maxInfected];
    
    // Initialize arrays containing the x and y positions of each infected person on this node.
    n = 0;
    int p;
    for (p = 0; p < personPortion; ++p) {
      MN[p].timeStep();
      if (MN[p].isInfected()) {
	x_pos[n]=MN[p].x;
	y_pos[n]=MN[p].y;
	n++;
      }
    }    
    // Mark remaining entries in array as null using -1.
    for (int j = n; j < maxInfected; j++){
      x_pos[j]=-1;
      y_pos[j]=-1;
    }

    // Set up an array on every node to hold the positions of all infected people.
    all_x = new int[nprocs*maxInfected];
    all_y = new int[nprocs*maxInfected];
    // Collect x and y position of all infected persons from all nodes. Allgather is a collective communication that performs a gather, then a scatter.
    MPI::COMM_WORLD.Allgather(x_pos,maxInfected,MPI::INT,all_x,maxInfected,MPI::INT);
    MPI::COMM_WORLD.Allgather(y_pos,maxInfected,MPI::INT,all_y,maxInfected,MPI::INT);

    // Create multiple threads to handle data.
#pragma omp parallel for default(shared) private(p) 
    for (p = 0; p < personPortion; ++p) {  
      // For each person on this node..   
      if (MN[p].isSusceptible()) {
	// If they are susceptible...
	for (int q =0 ; q < nprocs*maxInfected; ++q) {
	  // Loop through ALL infected people...
	  if (all_x[q] != -1) {
	    // If the infected person is within the transimission radius of the susceptible person...
	    if (dist(MN[p],all_x[q],all_y[q]) < Influenza.radius) {
	      // Simulate the chance of contracting the disease using a random number...
	      if (((float)(rand() % 100))/100 < Influenza.contagiousness) {
		// If transmission occurs, infect the susceptible person
		MN[p].infectedWith(Influenza);
		n++;
	      }
	    }
	  }
	}
      }
    }
    infectedPortion = n;
  } 

  ////////////// END OF SIMULATION ////////////////

  //Count final conditions

  int numInfected = 0, numSusceptible = 0;
  int p;
  for (p = 0; p < personPortion; ++p) {
    if (MN[p].isInfected())
      ++numInfected;
    if (MN[p].isSusceptible())
      ++numSusceptible;
  }
  int totalInfected = 0, totalSusceptible = 0;
  MPI::COMM_WORLD.Reduce(&numInfected,&totalInfected,1,MPI::INT,MPI::SUM,head);
  MPI::COMM_WORLD.Reduce(&numSusceptible,&totalSusceptible,1,MPI::INT,MPI::SUM,head);
  
  //Print summary of final conditions
  if (my_rank == head) {
  cout << numSusceptible << " persons are still susceptible" << endl;
  cout << numInfected << " persons are currently infected" << endl;
  cout << numPersons - numSusceptible - numInfected << " persons have recovered." << endl;
  }
  
  MPI::Finalize();
}
