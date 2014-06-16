Choosing the right Dimensions
=============================

One of the most important elements of CUDA programming is
choosing the right grid and block dimensions for the 
problem size. However it's not always clear which dimensions
to choose so we created an expiriment to answer the following
question: What effect do the grid and block dimensions have 
on execution times?

To answer this questions, we wrote a 
:download:`script <testMandelbrot.sh>` to run our mandelbrot code
for every 
block size between 1 and 512 and every number of threads 
per block between 1 and 512 which produced 262,144 data
points. We chose these ranges because our picture is
512x512 so each thread will calculate the value of at least
one pixel.

The device we ran the tests on was a Jetson TK1 which has
one Streaming Multiprocessor with 192 CUDA cores. To ensure
that our code was the only thing running on the GPU, we
first disabled the X server.

Results
#######

This is a 3D graph of our :download:`results <results.txt>`
where the z axis is the
log\ :sub:`2`\ (time) we took the log so that all results
fit neatly on the graph.

.. figure:: MediumPlot.png
    :align: center
    :figclass: align-center
    :width: 768
    :height: 510
    :alt: Execution time

There are a number of interesting things to note about this
graph:

- Trials with one block and many threads are faster than
  trials with many blocks of one thread each.

- There are horizontal lines indicating a spike in execution
  time after every 32 threads per block

- 512 threads per 512 blocks was the fastest execution time

- There are convex lines running through the middle of the 
  graph

Each of these observations relates directly to CUDA's
architecture or the specifics of the code. 

Many threads in 1 block is always faster than many blocks of one thread because only one block at a time can be run
on any given CUDA core, therefore when the blocks are divided
into warps, there is only one thread per core versus 32
threads per core.

Warp size also explains the horizontal lines every
32 threads per block. When block are are evenly divisible
into warps of 32, each block uses the full resources of the
CUDA cores on which it is run, but when there are (32 * x) + 
1 threads, every block uses
a full CUDA core to run only one thread, which accounts for 
the slowdown.

512x512 is the fastest execution time even though the GPU
can't run that many threads at a time. This is because 
it is inexpensive to create threads on a CUDA card.
cards and having one pixel per thread allows the GPU to
most efficently schedule warps as the CUDA cores become free.

The convex lines appear for a few different reasons. The
first has to do with our code. When the picture is evenly
divisible by the total number of threads and blocks, each 
thread performs the same amount of work so the CUDA cores
don't have to wait for threads calculating extra pixels. 
Second, when block and grid dimensions are about roughly 
equal, the block and warp schedulers share the work of 
dividing the threads.

CUDA best practices
###################

From these results we can draw up a list of best practices:

#. Try to make the number of threads per block a multiple of 32

#. Keep the number of threads per block and the number of blocks as close to equal as you can without violating the first tip

#. Keep the amount of work each thread does constant, it's inefficent to have one thread perform calculations for two pixels while the rest only calculate one.

#. When in doubt use more threads not less, creating threads is inexpensive.
