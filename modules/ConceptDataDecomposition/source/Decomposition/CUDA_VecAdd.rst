========================================
Vector Add with CUDA
========================================

Using the CUDA C language for general purpose computing on GPUs is well-suited to the vector addition problem, though there is a small amount of additional information you will need to make the code example clear.  On GPU co-processors, there are many more cores available than on traditional multicore CPUs. Each of these cores can execute a *thread* of instructions indicated in the code. Unlike the threads and processes of OpenMP and MPI, CUDA adds extra layers of organization of the threads: programmers set up *blocks* containing a certain number of threads, and organize the blocks into a *grid*.

For the problem we originally posed, where we had 8 elements in an array, one possible organization we could define in CUDA would be 4 blocks of 2 threads each, like this:

.. figure:: CUDABlocksThreads.png
    :width: 700px
    :align: center
    :alt: Decompose using 4 blocks of 2 threads each
    :figclass: align-center

CUDA convention has been to depict treads as squiggly lines with arowheads, as shown above. In this case, the 4 blocks of threads that form the 1-dimensional grid become analogous to the processing units that we would assign to the portions of the array that would be run in parallel.  In contrast to OpenMP and MPI, however, the individual threads within the blocks would each work on one data element of the array [#]_.

Determining the thread number
==============================

In all parallel programs that define one program that all threads will run, it is important to know which thread we are so that we can calculate what elements of the original array are assigned to us.  We saw with the MPI program, we could determine what the 'rank' of the process was that was executing.  In the OpenMP example in the previous section, thread number assignment to sections of the array in the for loop was implicit [#]_. 
In CUDA programs, we always determine which thread is running and use that to determine what portion of data to work on. The mapping of work is up to the programmer.

In CUDA programs, we set up the number of blocks that form the grid and we define how many threads will be used in each block. Once these are set, we have access to variables supplied by the CUDA library that help us define just what thread is executing. The following diagram shows how three of the available variables, corresponding to the grid, blocks, and threads within the blocks would be assigned for our above example decomposition:

.. figure:: CUDABThreadNumCalc.png
    :width: 700px
    :align: center
    :alt: blockDim.x = 1, threadId.x = 0,1 within each block, blockId.x = 0, 1, 2, 3
    :figclass: align-center
    

In this case of 1-dimensional arrays whose elements are being added, it is intuative to use a 1-dimensional grid of blocks, each of which has a 1-dimensional set of threads within it. For more complicated data, CUDA does let us define two or three-dimensional groupings of blocks and threads, but we will concentrate one this 1-dimensional example here.  In this case, we will use the 'x' dimension provided by CUDA variables (the 'y' and 'z' dimensions are available for the complex cases when our data would naturally map to them). The three variables that we can access in CUDA code for this example shown above are:

#. ``threadIdx.x`` represents a thread's index along the x dimension within the block.
#. ``blockIdx.x`` represents a thread's block's index along the x dimension within the grid.
#. ``blockDim.x`` represents the number of threads per block in the x direction.

In our simple example, there are a total of eight threads executing, each one numbered from 0 through 7. Each thread executing code on the GPU can determine which thread it is by using the above variables like this:

.. code-block:: c

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

.. [#] Each GPU, depending on what type it is, can run a certain number of CUDA blocks containing some number of threads concurrently.  Your code can define the number of blocks and the number of threads per block, and the hardware will run as many as possible concurrently.  The maximum number of threads you can declare in a block is 1024, while the number of blocks you can declare is a very large number that depends on your GPU card.  Here we show a simple case of four blocks of 2 threads each, which in CUDA terminology forms a grid of blocks.  This particular grid of blocks would enable each element in an array of size 8 to be computed concurrently in parallel by 8 CUDA threads. To envision how the decomposition of the work might occur, imagine that the block of yellow threads corresponds to processing unit 0; similarly for each additional block of threads.

.. [#] This isn't always the case in OpenMP programs-- you can know just what you are and execute a section of code accordingly.


The Host and the Device
========================

CUDA programs execute code on a GPU, which is a co-processor, because it is a device on a card that is separate from the motherboard holding the CPU, or central processing unit.  In CUDA programming, the primary motherboard with the CPU is referred to as the *host* and the GPU co-processor is usually called the *device*.  The GPU device has separate memory and different circuitry for executing instructions. Code to be executed on the GPU must be compiled for its instrution set.

CUDA Code for Vector Add
=========================

The overall structure of a CUDA program that uses the GPU for computation is as follows:

1. Define the the code that will run on the device in a separate function, called the *kernel* function.
2. In the main program running on the host's CPU:
    a. allocate memory on the host for the data arrays.
    #. initialze the data arrays in the host's memory.
    #. allocate separate memory on the GPU device for the data arrays.
    #. copy data arrays from the host memory to the GPU device memory.
3. On the GPU device, execute the *kernel* function that computes new data values given the original arrays. Specify how many blocks and threads per block to use for this computation.
4. After the *kernel* function completes, copy the computed values from the GPU device memory back to the host's memory.

Here is the actual code example for 8-element vector addition, with these above steps numbered. This is in the file
**VectorAdd/CUDA/VA-GPU-simple.cu** in the compressed tar file of examples that accompanies this reading.


.. literalinclude:: ../code/VectorAdd/CUDA/VA-GPU-simple.cu	
    :language: c
    :linenos:
  
CUDA development and Special CUDA Syntax
=========================================

The CUDA compiler is called **nvcc**.  This will exist on your machine if you have installed the CUDA development toolkit.

The code that is to be compiled by nvcc for execution on the GPU is indicated by using the keyword  ``__global__`` in front of the kernel function name in its definition, like this on line 21:

.. literalinclude:: ../code/VectorAdd/CUDA/VA-GPU-simple.cu	
    :language: c
    :lines: 21-22
    
The invocation of this kernel function on the GPU is done like this in the host code on line 82:

.. literalinclude:: ../code/VectorAdd/CUDA/VA-GPU-simple.cu	
    :language: c
    :lines: 82-83
    
Note how you set the number of blocks and the number of threads per block to use with the ``<<< >>>`` syntax.

Larger Problems
===============

There is an additional file in the tar archive supplied for this reading that is called
**VectorAdd/CUDA/VA-GPU-larger.cu**.  You could experiment with larger arrarys in this version of the code.  Comments within it explain how the thread assignment to array elements works if your array is larger than the number of total threads you use.  CUDA and GPUs are better suited to much larger problems than we have shown for illustration here.
