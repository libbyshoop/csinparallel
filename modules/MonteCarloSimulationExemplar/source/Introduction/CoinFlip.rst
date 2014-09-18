Testing out random number generators: Flip a coin many times
=============================================================

A simple way to see how well a random number generator is working is
to simulate flipping a coin over and over again for many trials.

Let's look at some C/C++ code to do this.  The listing below
shows how we can use srand() to seed our random number generator with
a large integer and then make many calls to rand() (or rand_r() on linux/unix)
to obtain a series
of random integers.  If the integer is even, we call it a 'head' coin flip, otherwise
it is a 'tail'.  This code sets up trials of coin flips with ever increasing
numbers of flips.  It also calculates the Chi Square statistic using the number of heads
and number of tails.  A rule of thumb in the case of heads and tails is that if the
Chi-Square value is around 3.8 or less, we have a good random distribution of the
even and odd values.  We want to verify that the random number generator provides
such an independent distribution.

.. seealso::
    For more details about chi square calculations and how they measure whether a set of values
    flows an independent distribution, please see
    `A Chi-square tutorial <http://www.radford.edu/~rsheehy/Gen_flash/Tutorials/Chi-Square_tutorial/x2-tut.htm>`_,
    which shows an example for coin-flipping.
    
    There are many other examples you can find by searching on the web.

In the main() there is a while loop that conducts the trials of coin flips.  Each trial is
conducted by obtaining random numbers in the for loop on line 60. 
You can download the file
:download:`coinFlip_seq.cpp <../code/montecarlo_openmp_cpp/coinFlip/coinFlip_seq.cpp>` and try this code below yourself.  You should note that the longer trials with many coin flips take a somewhat long time (on the order of 20 seconds, depending on your machine).

In the next section, we will look at parallelizing code using threads and OpenMP, then we will explore how we can conduct the coin-flipping simulation in parallel so that it runs considerably faster.

.. literalinclude:: ../code/montecarlo_openmp_cpp/coinFlip/coinFlip_seq.cpp	
    :language: c
    :linenos:
    