Monte Carlo Simulations on Other Parallel and Distributed Platforms
====================================================================

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
You can get more information about the CUDA random library at:
http://docs.nvidia.com/cuda/curand/introduction.html We allocate space on the GPU for a
random number generator for each thread and the result:

::

    cudaMalloc((void **)&devStates, sizeof(curandState)*BLOCK_SIZE);
    cudaMalloc((void **)&dev_result, sizeof(unsigned int));
    cudaDeviceSynchronize();


We define a new function, run_simulations(), which runs the simulations on the GPU:

.. literalinclude:: drawFourSuitsCuda.cu
   :language: c++
   :lines: 54-86

From this function, we call the other helper function, so we add the __device__ 
directive to them to allow them to be called on the GPU:


Here is the source code for this section:

:download:`drawFourSuitsMPI.cpp <drawFourSuitsMPI.cpp>`
:download:`drawFourSuitsCuda.cu <drawFourSuitsCuda.cu>`