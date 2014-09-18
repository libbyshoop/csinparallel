Drawing Four Cards of the Same Suit
===================================
.. image:: RoyalFlush.jpg
    :width: 200
    
Now let's turn our attention to the card game of Poker.
There are methods to calculate the probability of drawing the various types of hands (see the `Wikipedia Poker Probability Page <http://en.wikipedia.org/wiki/Poker_probability>`_ for explanation).
For our next example, we will examine one such type of hand with the following question:

*If you are dealt a random hand of 5 cards, what is the probability that four of the cards each have a different
suit?*

To answer this question, we simulate shuffling a deck of cards and drawing a hand of cards.


Code Files
----------

For this code, we have separate versions for Windows, which uses rand(), and linux, which uses rand_r() as the random number generators.

===========================   ======================================================================================================
Linux
===========================   ======================================================================================================
     sequential version:      :download:`drawFourSuits_seq.cpp <../code/montecarlo_openmp_cpp/drawFour/Linux/drawFourSuits_seq.cpp>`
     OpenMP version:          :download:`drawFourSuits_omp.cpp  <../code/montecarlo_openmp_cpp/drawFour/Linux/drawFourSuits_omp.cpp>`
===========================   ======================================================================================================

===========================   ======================================================================================================
Windows
===========================   ======================================================================================================
     sequential version:      :download:`drawFourSuits_seq.cpp <../code/montecarlo_openmp_cpp/drawFour/Windows/drawFourSuits_seq.cpp>`
     OpenMP version:          :download:`drawFourSuits_omp.cpp  <../code/montecarlo_openmp_cpp/drawFour/Windows/drawFourSuits_omp.cpp>`
===========================   ======================================================================================================


Sequential code
---------------

We represent the deck of cards as an array of integers. Our function for simulating
deck shuffling is not the most efficient, but it tries to capture how a traditional
"fan" shuffle actually works. We also have helper functions for initializing a deck,
drawing a hand, and checking if the hand has four cards of the same suit. Download the
appropriate sequential code file for your environment and study it.  Note all the places
where random numbers are generated for two aspects of the problem: shuffling the deck and
picking cards from the deck to form a hand.

Using these helper functions, it was straightforward to write testOneHand,
which initializes a deck, shuffles it, draws a hand, and then checks if
all four suits are represented.

.. literalinclude:: ../code/montecarlo_openmp_cpp/drawFour/Linux/drawFourSuits_seq.cpp
   :language: c++
   :lines: 198 - 210

Open MP Version
---------------

Converting our sequential code to use OpenMP is quite simple. We add a pragma 
compiler directive to the main simulation loop to run the loop simultaneously
on multiple CPUs. The directive tells OpenMP to give each thread a different
copy of i since each thread needs to keep track of its own loop iterations. 
numTests is shared because the total number of tests to run is doubled only
once per iteration of the out while loop. (If each thread doubled it, we would
go up by more than a factor of two.) Finally, the directive `reduction (+:total)`
tells OpenMP to combine each of the threads' partial results by summing to find
the total number of hands that contained all four suits. 

.. literalinclude:: ../code/montecarlo_openmp_cpp/drawFour/Linux/drawFourSuits_omp.cpp
   :language: c++
   :lines: 75-92

Note that the above example is for the linux version of the code, which uses the thread-safe rand_r() function.

