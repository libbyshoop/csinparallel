Roulette Simulation
===================

Our next is example is a Roulette simulation. We have a main simulation
loop that is similar to the previous example. Notice that we use a OpenMp
directive to seed the random number generator separately in each thread (see
the discussion about randomness in the Introduction.) 

.. literalinclude:: rouletteSimulation.cpp
   :language: c++
   :lines: 32-72

The actual simulation function is getNumWins and is quite simple. As before,
it runs the simulations in parrallel in a loop. numSpins and myBet are shared
between theads while spin is the loop index and unique to each thread. Again,
analogously to the previous example, we combine the partial results from each 
threads with 'reduction(+:wins)'.

.. literalinclude:: rouletteSimulation.cpp
   :language: c++
   :lines: 76-96

The function that actually runs a single simulation of the Roulette wheel is quite
simple. It simply generates a random number to represent the slot that the ball
ends up in and gives a payout according to the rules of Roulette.

.. literalinclude:: rouletteSimulation.cpp
   :language: c++
   :lines: 130-145

Todo
----

Now see if you can modify the provided code to use MPICH and/or CUDA instead of 
of OpenMP. If you are feeling really ambitious, you can even make the program use
hybrid parallelism (ie use MPICH to divide the problem between a cluster and OpenMP
to divide each node's problem into threads). 

Here is the full source code for the OpenMp version:
:download:`rouletteSimulation.cpp <rouletteSimulation.cpp>`

