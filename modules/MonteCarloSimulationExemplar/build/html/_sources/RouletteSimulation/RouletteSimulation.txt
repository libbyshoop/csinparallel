.. figure:: 640px-Sahara_Hotel_and_Casino_2.jpg
    :alt: Roulette Wheel Picture
    
    "Sahara Hotel and Casino 2" by Antoine Taveneaux - Own work. Licensed under Creative Commons
    Attribution-Share Alike 3.0 via
    `Wikimedia Commons <http://commons.wikimedia.org/wiki/File:Sahara_Hotel_and_Casino_2.jpg#mediaviewer/File:Sahara_Hotel_and_Casino_2.jpg>`_

Roulette Simulation
===================

An American Roulette wheel has 38 slots: 18 are red, 18 are black, and 2 are
green, which the house always wins. When a person bets on either red or black,
the odds of winning are 18/38, or 47.37% of the time.

Our next is example is a simulation of spinning the Roulette wheel. We have a
main simulation loop that is similar to the coin-flipping example. The code
for determining a win on each spin is more involved than flipping a coin, and
the sequential version,
:download:`rouletteSimulation_seq.cpp <../code/montecarlo_openmp_cpp/roulette/rouletteSimulation_seq.cpp>`
is decomposed into several methods. Look at this original code file to see how
we run the simulations using increasing numbers of random spins of the wheel.

The function that actually runs a single simulation of the Roulette wheel, called spinRed(),  is quite
simple. It generates a random number to represent the slot that the ball
ends up in and gives a payout according to the rules of Roulette.

.. literalinclude:: ../code/montecarlo_openmp_cpp/roulette/rouletteSimulation_seq.cpp
   :language: c++
   :lines: 100-118

.. note:: The sequential version of the simulation takes a fair amount of time. Note how long.
    Also note how many simulated random spins it takes before the distribution of spins
    accurately reflects the house odds.

Parallelism to the Rescue
--------------------------

We add OpenMP parallelism as in the coinFlip example, by running the loop of random spins
for each trial on several threads. This code is in this file that you can download:
:download:`rouletteSimulation_omp.cpp <../code/montecarlo_openmp_cpp/roulette/rouletteSimulation_omp.cpp>`
The actual simulation function is getNumWins(): 

.. literalinclude:: ../code/montecarlo_openmp_cpp/roulette/rouletteSimulation_omp.cpp
   :language: c++
   :lines: 103-124

Notes about this code: numSpins and myBet are shared
between threads while spin is the loop index and unique to each thread.
When using rand_r() as the thread-safe random number generator in linux/unix,
the seed should be private to each thread also.
Like the previous example, we combine the partial results from each 
thread with `reduction(+:wins)`.




