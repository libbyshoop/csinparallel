**************************
Time Trials and Evaluation
**************************

After implementating the algorithms described in the previous exercises, we ran time trials to compare them. Here are the results, presented for you to compare yours.

All tests were run using 200 iterations of the simulation (which corresponds to 50 days). The general type of epidemic simulation runs until there are no more infected people, but we wanted to be able to compare speed-up so we needed to standardize the amount of computation.

:Simulation parameters:
	
	- ``width = 10000 //width of the 'world'``
	
	- ``height = 10000 //height of the 'world'``
	
	- ``numPersons = 20000 //total population``
	
	- ``initialInfected = 200 //initial number of infected people``
	
	- ``numIterations = 200 //the number of six-hour time periods to be simulated``


:Infection parameters:
	
	- ``duration = 28 //number of 6 hour periods a person remains infected``
	
	- ``radius = 45 //the size of the area of possible transmission``

	- ``contagiousness = .5 //the probability of transmission``

Results
#######

**Serial:** On average, the serial code took 1:15-1:20.

**OpenMP:** There was drastic speedup when using OpenMP, which is not surprising since the serial simulation relies heavily on a computationally-intensive for loop. The OpenMP version took seven to ten seconds, about a tenfold speedup.

**MPI:** On average, the MPI code took thirty seconds with four cores and twenty seconds with 32 cores. For even larger systems, we think the MPI version will eventually achieve better results than OpenMP, but the testing we did never reached that point.

**Hybrid (MPI/OpenMP):** Using four cores, the hybrid code ran for forty seconds. This is not as drastic of a speedup as going from serial to OpenMP because the number of people on each process is smaller (since they are divided up among processes). For large, real-world simulations where the number of people on each process is very large (for example, considering total populations in the millions) the hybrid is most likely the best parallelization to use: the overhead from the additional OpenMP will be insignificant.

**Go:** This program using the same parameters ran in ten to thirteen seconds, astonishing for such a readable, high-level language. The program's blocking call may also have wasted time. Additionally, Go code was run on a MacBook Pro with a 2.53 GHz Intel Core i5 processor, whereas other implementations were run on a machine with a 2.66 GHz Intel Quad Core processor - run on the same machine, the OpenMP and Go programs would likely run in very comparable time.

Evaluation
##########

.. comment
	Flesh out so that it reads more like the evaluation section at the end of Drug Design Exemplar (i.e., discussion of patterns, etc)?

The contrast of the serial to OpenMP is impressive for only adding 4 additional lines of code. The MPI versions rely on collective communication including the Allgather function as another gather-scatter programming pattern.

Using Go was an interesting approach, and channels offer a different programming pattern than those studied with the other languages. Impressively, it is as fast as OpenMP.

The algorithm is a fairly intuitive representation of our natural understanding of how disease spreads. The simplifications made created an algorithm that was geared to computer scientists and not mathematicians.