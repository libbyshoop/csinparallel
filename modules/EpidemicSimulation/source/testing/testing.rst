**************************
Time Trials and Evaluation
**************************

After implementating the algorithms described in the previous exercises, we ran time trials to compare them. Here are the results, presented for you to compare yours.

All tests were run using 200 iterations of the simulation (which corresponds to 50 days). The general type of epidemic simulation runs until there are no more infected people, but we wanted to be able to compare speed-up so we needed to standardize the amount of computation.

Parameters:
width = 10000 //width of the 'world'
height = 10000 //height of the 'world'
numPersons = 20000 //total population
initialInfected = 200 //initial number of infected people
numIterations = 200 //the number of 6 hour time periods to be simulated

Influenza parameters:
duration = 28 //number of 6 hour periods a person remains infected
contagiousness = .5 //the probability of getting sick from the interaction of a sick person with an infected person
radius = 45 //the size of the area of possible transmission


Serial: On average, the serial code took 1:15-1:20.

OpenMP: There was drastic speedup when using OpenMP which is not surprising since the serial simulation relies heavily on a computationally-intensive for loop. The OpenMP version took :07-:10 seconds which is around a 10 fold speed-up.

MPI:  On average, the MPI code took :30 seconds with 4 cores and :20 seconds with 32 cores. For even larger systems, we think the MPI version will eventually achieve better results than OpenMP, but the testing we did never reached that point.

Hybrid (MPI/OpenMP): Using 4 cores the Hybrid code ran for :40 seconds. This is not as drastic of a speedup compared to going from serial to OpenMP since the number of people on each process is smaller (since they are divided up among processes). For large, real-world simulations where the number of people on each process is very large (for example considering total populations in the millions) the Hybrid is most likely the best parallelization to use since the overhead from the additional OpenMP will be insignificant.

Go language: This program using the same parameters ran in 10 -13 seconds. This is astonishing given a program that reads like python. This program also has a blocking call which may waste time.

Note that this code was not run on the lab machine but on a MacBook Pro with a 2.53 GHz Intel Core i5 processor. Since the lab machines (where the OpenMP program ran in 7-10 seconds) have a slightly faster processor (2.66 GHz Intel Quad Core), I would call consider this programs comparable and competitive with each other.


Conclusions
###########

This programming exercise shows more variation in speedup than other labs used with the PDC class. The contrast of the serial to OpenMP is impressive for only adding 4 additional lines of code. The MPI versions rely on collective communication including the Allgather function as another gather-scatter programming pattern.

Using Go was an interesting approach, and channels offer a different programming pattern than those studied with the other languages. And better yet, it is as fast as OpenMP.

The algorithm is fairly intuitive to the natural understanding of how disease spreads. The simplifications we made created an algorithm that was geared to computer scientists and not mathematicians.
