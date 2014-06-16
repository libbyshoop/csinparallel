Choosing the right Dimensions
=============================

One of the most important elements of CUDA programming is
choosing the right number of blocks and threads for the 
problem size. We must first answer the question, what effect
does the CUDA architectue have on execution times for problems
with different thread dimensions?

To answer these questions, we wrote a 
:download:`script <test.sh>` to run our code for every 
block size between 1 and 512 and every number of threads 
per block between 1 and 512 which produced 262,144 data
points. We chose these points because our picture is
512x512 with the largest dimensions there is one thread 
working per pixel.

The device we ran the tests on was a Jetson TK1 which has
one Streaming Multiprocessor with 192 CUDA cores. To ensure
that our code was the only thing running on the Jetson, we
first disabled the X server.

Results
#######

We created a 3D graph of our results where the z axis is the
log\ :sub:`2`\ (time) (we took the log so that all results
fit neatly on the graph).

.. figure:: 


There are a number of interesting things to note about this
graph:

- There are convex lines running through the middle of the 
  graph

- There are spikes in execution time after every 32 threads
  per block

- 512 threads per 512 blocks was the fastest execution time

- x threads in 1 block is faster than 1 thread in x blocks

Each of these observations relates directly to CUDA's
architecture and the specifics of the code. 

x threads in 1 block is always faster than 1 thread in x
blocks. This is because only one block at a time can be run
on any given CUDA core, therefore when the blocks are divided
into warps, there is only one thread per core versus 32
threads per core when there is only one block.

Warp size also explains the horizontal lines ever multiple of
32 threads per block. When threads per block is a multiple of
32, each block uses the full resources of any given CUDA core
, but when there are (32 * x) + 1 threads, every block uses
a full CUDA core to run only one thread, which accounts for 
the slowdown.

512x512 is the fastest execution time even though the GPU
can't runn that many threads at a time. this is because 
creating threads is a very inexpensive operation on CUDA
cards and having one pixel per thread allows the GPU to
most efficently schedule it's resources.

The convex lines appear for a few different reasons. The
having to do with our code. When the picture is evenly
divisible by the total number of threads and blocks, each 
thread performs the same amount of work. Secondly when block
and grid dimensions are about roughly equal, the block
and warp schedulers share the work of dividing the threads
rather than counting exclusively on one of them.

CUDA best practices
###################

From these results we can draw up a list of best practices:

#. Try to make the number of threads per block a multiple of 32

#. Keep the number of threads per block and the number of blocks as close to equal as you can without violating the first tip

#. Keep the amount of work each thread does constant, it's inefficent to have one thread perform calculations for two pixels while the rest only calculate one.

#. When in doubt use more threads not less, creating threads is inexpensive.
