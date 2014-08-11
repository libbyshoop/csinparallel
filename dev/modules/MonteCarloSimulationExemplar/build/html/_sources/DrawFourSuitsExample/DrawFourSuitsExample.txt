Drawing Four Cards of the Same Suit
===================================

For our first example, we are interested in the following question: If you are dealt 
a random hand of 5 cards, what is the probability that for of the cards have the same
suit? To answer this question, we simulate shuffling a deck of cards and drawing a hand of cards. 

Sequential code
---------------

We represent the deck of cards as an array of integers. Our function for simulating
deck shuffling is not the most effieicent, but it tries to capture how a traditional
"fan" shuffle actually works. We also have helper functions for initializing a deck,
drawing a hand, and checking if the hand is a flush. 

.. literalinclude:: drawFourSuitsSequential.cpp
   :language: c++
   :lines: 91-241

Using these helper functions, it's straightforward was straightforward to write testOneHand,
which initializes a deck, shuffles it, draws a hand, and then checks if it is a flush. 

Open MP Version
---------------

Converting our sequential code to use OpenMP is quite simple. We add a pragma 
compiler directive to the main simulation loop to run the loop simultaneously
on multiple CPUs. The directive tells OpenMP to give each thread a different
copy of i since each thread needs to keep track of its own loop iterations. 
numTests is shared because the total number of tests to run is doubled only
once per iteration of the out while loop. (If each thread doubled it, we would
go up by more than a factor of two.) Finally, the directive 'reduction (+:total)'
tells OpenMP to combine each of the threads' partial results by summing to find
the total number of hands that were flushes. 

.. literalinclude:: drawFourSuitsOpenMP.cpp
   :language: c++
   :lines: 61-77

MPICH Version
-------------

Converting the sequential code to use MPICH instead is only slightly more difficult
than using OpenMp. First we have to initialize MPI and calculate the number of tests
each instance needs to run:
::

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ...
    testsToRun = numTests/size;
    if(rank == size - 1) testsToRun += numTest % size //assign remaining tests to last worker

Then, each instance runs a separate simulation loop as before. The other difference
is we need to explictly send the partial results from each worker to the Master instance.
The Master node sums the partial results and displays the answer:

.. literalinclude:: drawFourSuitsMPI.cpp
   :language: c++
   :lines: 92-105

Cuda Version
------------

The changes needed to run the code with CUDA are slightly more involved. We have several
issues we have to deal with. First, we have to copy our results to and from the the device.
The other issue is that we can't use the standard random library from the device. 
Fortunately, Nvidia provides build in libraries for generating random numbers on a GPU.
You can get more information about the CUDA random library at: http://docs.nvidia.com/cuda/curand/introduction.html We allocate space on the GPU for a random number generator for
each thread and the result:
::

    cudaMalloc((void **)&devStates, sizeof(curandState)*BLOCK_SIZE);
    cudaMalloc((void **)&dev_result, sizeof(unsigned int));
    cudaDeviceSynchronize();


We define a new function, run_simulations that runs simulations on the GPU:

.. literalinclude:: drawFourSuitsCuda.cu
   :language: c++
   :lines: 54-86

From this function, we call the other helper function, so we add the __device__ 
directive to them to allow them to be called on the GPU:

Here is the source code for this section:
:download:`drawFourSuitsSequential.cpp <drawFourSuitsSequential.cpp>`
:download:`drawFourSuitsOpenMP.cpp <drawFourSuitsOpenMP.cpp>`
:download:`drawFourSuitsMPI.cpp <drawFourSuitsMPI.cpp>`
:download:`drawFourSuitsCuda.cu <drawFourSuitsCuda.cu>`

