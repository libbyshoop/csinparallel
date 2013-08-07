Introduction to the Game of Life
================================

Conway’s Game of Life is a well-known example of a cellular automaton. A
cellular automaton consists of a regular, uniform lattice of sites (or
cells), where each site takes on one value of a discrete variable. This
lattice can potentially be any number of dimensions. The state of the
automaton changes at discrete time steps. For each step of time, the
state of each cell is changed by following a set of rules that determine
the state of that cell in next generation, based on its current state
and the current state of some set of cells defined to be its neighbors.
This is done for every cell in the grid, conceptually at the same time.

The Game of Life was designed by British mathematician John Conway in
1970. It is a two-dimensional cellular automaton with very simple rules.
The cells are square, and the grid is considered to be infinite. Each
cell has 8 neighbors surrounding it. For each generation, the following
table of rules is considered to detmine the state of each cell in the
next generation:

Rules for Conway’s Game of Life

+------------+---------+-----------+----------------+----------------+-----------+
| Number of living neighbors                                                     |
+============+=========+===========+================+================+===========+
|            |         |<2         |2               |3               |>3         |
+------------+---------+-----------+----------------+----------------+-----------+
|Cell State  | *Alive* |Die        |Continue to live|Continue to live|Die        |
+            +---------+-----------+----------------+----------------+-----------+
|            | *Dead*  |Remain dead|Remaint dead    |Come to life    |Remain Dead|
+------------+---------+-----------+----------------+----------------+-----------+

This might be expressed in pseudocode as follows::

   ALGORITHM NextGeneration(celL)
   --------------------------------------------------
   num_neighbors ← cell.COUNT_NUM_NEIGHBORS()
   if cell.ISALIVE() then
      if num_neighbors < 2 or num_neighbors > 3 then
         cell.setState(DEAD)
      endif
   else
      if num_neighbors = 3 then
         cell.setState(ALIVE)
      endif
   endif

The rules of Life are simple, yet they were chosen carefully. While some
cellular automata are not very exciting because their cells all die out
very quickly, or their alive cells grow explosively forever, Conway’s
Game of Life is interesting because its rules allow for a wide range of
behavior. There can be steady-state sets of cells (still lifes),
oscillating sets of cells (oscillators), and moving sets of cells
(spaceships), as well as complex changes that occur when these patterns
interact.

The Game of Life has seen much interest over the years because despite
its simple rules, it is possible to find very complex patterns. In fact,
Life is Turing complete, so given enough time and a Life grid set up in
an appropriate initial state, it is theoretically possible to compute
anything that can be computed. In 2000, a Turing Machine was implemented
in Life, and in 2010, a Universal Turing Machine was implemented. Also
in 2010, a pattern that constructs a copy of itself was invented.
However, it destroys itself in the process, so “real” artificial life
that can replicate itself multiple times has yet to be created.

Algorithm
*********

The most straightforward algorithm for Life on today’s real computers is
to store the grid in memory with alive cells as bytes set to 1 and dead
cells as bytes set to 0. Two copies of the grid would be required so
that the next generation could be calculated from the previous
generation without disrupting the values needed for the calculation.

Since the Life grid is supposed to be infinite, this method of
representing the grid has to restrict the grid size. Ideally, these
limits would be set to an appropriate value for the pattern so that its
development is not disrupted by reaching the boundaries.

A C implementation of this algorithm might look like this:

serial.tex

The code takes in a parameter grid\_in, which is a 2D byte array that
contains the current generation. width and height are the dimensions of
this array. The next generation is written into the array grid\_out. For
simplicity, the code simply does not compute the cells on the
boundaries. (Another way of handling the boundaries would be to have
them wrap around so that cells going off one boundary would appear on
the other side.)

Parallelization
***************

It is easy to see how the computation of Life using the above algorithm can be parallelized. In an *m* by *n* grid, *m x n* calculations need to be performed, but none of them are dependent on each other. Therefore, theoretically there could be up to *m x n* processors used to perform the calculation, completely in parallel.

There are six different ways to compile and run this Game of Life code:

(1) Traditional single-process implementation; no communication or synchronization is done.

(2) Shared memory implementation using OpenMP. OpenMP is used to create a number of threads equal to the number of cores on the node and parallelize a handful of for loops.

(3) Distributed implementation using MPI. Each MPI process has its own grid, sharing left and right boundaries with neighboring processes.

(4) GPU programming implementation using CUDA. The Life grids can be created in the regular system memory, and then copied to a CUDA-capable device using the CUDA API. Then, function would run code on the GPU.

(5) Hybrid implementation, combining OpenMP and MPI.

(6) Hybrid implementation, combining CUDA and MPI.

Compiling
*********

A Makefile is included with this project to ease the compiling process. In all cases, a binary named "Life" will be created.

By default, running "make" with no options will compile all known implementations with full graphical support via X11 libraries. These can be compiled individually by their name, which can be found by running `make help`. This will print the list of possible targets, each of which builds a binary named Life.XXX where XXX is the name of the target.

Additionally, graphical support can be completely disabled by running `make NO_X11=1` plus any of the desired targets.

Options
*******

This version of Life takes the following arguments::

   * -c | --columns number: Number of columns in grid
   * -r | --rows number: Number of columns in grid
   * -g | --gens number: Number of generations to run
   * -i | --input filename: Input file
   * -o | --output filename: Output file
   * -h | --help: This help page
   * -t[N] | --throttle[=N]: Throttle graphical display (N generations per second)
   * -x | --display: Use a graphical display
   * --no-display: Do not use a graphical display

If none of these are given, the following defaults are used:

* Rows:        105
* Columns:     105
* Generations: 1000
* Input file:  None
* Output file: None
* Throttle:    No
* Display:     Yes

Note: --throttle has no effect if --no-display is given or NO_X11 is specified at compile time. If --throttle or -t are given without a value, a default value of 60 generations/second will be used.

The input file should contain space-separated values, two per line. The first line specifies the dimensions, columns and rows respectfully. Each subsequent line gives the column,row for a living cell. Order has no effect. Any coordinate not given is assumed to contain a dead cell.

For example, a 5x5 grid with three living cells::

   5 5
   1 2
   2 2
   3 5

Running
*******

Each implementation is run a little differently::

   Serial:
      $ ./Life.serial

   OpenMP:
      $ ./Life.c-openmp

   MPI: (where <N> is the desired number of processes)
      $ mpirun -x DISPLAY=node000:0.0 -machinefile machines -np <N> ./Life.c-mpi

   CUDA:
      $ ./Life.c-cuda

   Hybrid MPI/OpenMP: (where <N> is the desired number of processes)
      $ mpirun -x DISPLAY=node000:0.0 -machinefile machines -np <N> ./Life.c-mpi-openmp

   Hybrid MPI/CUDA: (where <N> is the desired number of processes)
      $ mpirun -x DISPLAY=node000:0.0 -machinefile machines -x LD_LIBRARY_PATH -np <N> ./Life.c-mpi-cuda

OpenMP
------

If coding in C, C++, or Fortran, it is easy to parallelize the above
code listing by adding an OpenMP directive that directs the compiler to
parallelize the loop. In C, it would look like this:

openmp.tex

At the point in the program where the pragma is placed, a default number
of threads (usually the number of processors available on the machine)
is created, and the iterations of the loop are divided up evenly among
them. No synchronization is needed except for the implicit barrier at
the end of the for loop.

If multiple generations are to be computed, it would be more optimal to
create threads at a level before the compute\_next\_generation()
function, so that the threads could work on multiple generations with
lightweight synchronization between each generation before being
destroyed. This is what I did in my implementation.

GPU
---

As a highly data parallel problem, the computation of the next
generation of a Life grid is quite appropriate for implementation on a
graphics processing unit (GPU).

Performing general-purpose computations on GPUs is a fairly new,
developing area, and not many programs take advantage of it yet. One
framework that shows some promise is OpenCL, which is an open
specification that allows programs to execute on GPUs. It currently
works on both the latest ATI/AMD devices and Nvidia devices.

However, the more developed framework at this point is Nvidia’s CUDA
(Compute Unified Device Architecture). CUDA is the name of a parallel
computing architecture for Nvidia’s GPUs, and its software development
kit comes with an API (libraries and headers) and a compiler for a
language very similar to C in which functions to be executed on the GPU
can be written.

The Life grids can be created in the regular system memory, and then
copied to a CUDA-capable device using the CUDA API. Then, the
implementation of the compute\_next\_generation() function would run
code not on the CPU, but instead launch a “kernel” to be executed on the
GPU.

The following is a kernel for computing the next generation of a Life
grid:

cuda.tex

It it fairly similar to the C code, as the CUDA language is based on C.
The \_\_global\_\_ keyword indicates an entry point to a kernel.

The following code is the actual compute\_next\_generations() function
that is equivalent to the serial and OpenMP versions given earlier. It
is a wrapper around the kernel.

cuda-launch.tex

Perhaps the strangest thing about the CUDA code is that there is no for
loop. This is because if the kernel is called with the correct number of
thread blocks and threads per block, each data element will be assigned
to 1 thread. The calculation is expressed merely for 1 element, but this
computation is being done in a massively parallel manner by the
launching of the kernel with the appropriate number of thread blocks and
threads per block. This would be a very poor way to write the program on
a CPU. But on the GPU, writing the program in this way is possible, and
actually preferable, because the the CUDA architecture uses very
lightweight threads, implemented in hardware. Large numbers of these
threads can be executed at the same time, as the GPU consists of a
number of multiprocessors, each of which can command 32 threads
simultaneously.

In the above code, grid of Life cells is divided into thread blocks of
size 16 x 16 x 1. Therefore, each thread block contains 256 threads. As
the code is written, the grid dimension must be evenly divisible by 16
for all grid cells to be calculated. The first lines in the kernel have
the thread figure out which grid element it is responsible for; then,
the remainder of the kernel does that calculation.

Implementation
**************

The Platform
------------

The LittleFe platform provided a way for me to try implementing Conway’s
Game of Life using CUDA and OpenMP. The LittleFe is a “portable cluster”
that has 6 motherboards connected by an ethernet switch. Each node has a
CUDA-capable GPU and a dual-core “Intel Atom” processor. However,
despite the availability of 6 nodes, I only used one node because I was
not doing any distributed parallelism (e.g. with MPI). The operating
system is a 32-bit version of the Bootable Cluster CD, which is based on
Debian GNU/Linux.

The Code
--------

The implementation of Life that I wrote is interactive. The generations
are rendered using OpenGL, and the GUI uses the GLUT library, which is a
common library used for cross-platform OpenGL programs of low to medium
complexity. The arrow keys can be used to move around, the plus and
minus keys zoom in and out, respectively, and Space and ‘b’ speed up and
slow down the simulation, respectively. By default, it will start up
with the R Pentonimo pattern, which is a small but interesting pattern.
Give the –help argument to see all command line options. (A name of a
file containing a Life grid can be given, but there are only 2 Life file
formats supported, and not all files of those formats will open
correctly yet.) See the README in the code directory for a little more
information.

I designed the program so that, to some extent, different
implementations of the grid representation and computation could be
used. The files cuda.c and openmp.c contain the same externally visible
functions, such as a function to advance a number of generations in the
simulation, but their implementations are different. A CUDA-powered
executable and an OpenMP powered executable are built by linking the
rest of the object files with these different implementations.

The source code files are the following:

-  life.c

   , which contains main(); Creates the window and starts the event
   loop, or runs a timed test.

-  callbacks.c

   , which contains the callbacks for the GUI (display, key pressed,
   etc.).

-  cuda.c

   , which is the CUDA implementation of the computation.

-  cuda-kernel.cu

   , which goes along with cuda.c and consists of the code that launches
   the kernel, and the kernel itself.

-  openmp.c

   , which is the OpenMP implementation of the computation.

-  fileio.c

   , which currently can read in LIFE grids in the .rle (run length
   encoding) and .lif formats, which are two file formats for life
   patterns. I include a directory patterns that contains some patterns
   in these formats. (Note: I put together the file input functions
   quickly and they will not open every file correctly. All the ones in
   patterns/lifep that I have tried have worked, however).

As mentioned, the rendering is done using OpenGL. This is unlike the
Life program that was used in the example that came with the Bootable
Cluster CD, which used the Xlib XDrawRectagle() function to send the
data through the X server. I did not really use any of the code from the
BCCD version. In my version, I did something somewhat unusual by
treating the grid data as pixel data and passing it directly to OpenGL
using GLDrawPixels(), indicating that the data is in the
GL\_UNSIGNED\_BYTE format and contains only one color component. This
allows the pattern of dead and alive cells to be send directly to the
GPU for rendering without any extra processing.

In the case of the CUDA implementation, the data does not even leave the
GPU. I created an OpenGL buffer object with GLGenBuffers() and
GLBufferData(). This a region of memory allocated on the GPU, which
normally would be filled with static pixel or vertex data to be
repeatedly rendered without sending the data to the GPU over and over.
In this case, however, it is used for dynamic pixel data, as it doubles
as the actual representation of the grid. CUDA provides a way to
register OpenGL buffer objects and retrieve pointers to the memory
address of them in device memory, which can then be passed to a CUDA
kernel. Therefore, other than the picture that comes out of the graphics
card, the data is simply kept on the GPU the entire time and never
reaches the CPU.  [1]_ This avoids the overhead of sending large
quantities of data to the GPU over and over.

My program does not yet provide a way to calculate an arbitrary Life
generation and dump the output to a file. If it was to do this, it would
be nice to have an option to do so without initializing the GUI. The
program does, however, provide a way to do timing, as described in the
next section.


.. [1]
   When I was testing the program, I used the LittleFe remotely.
   Normally it is not possible to run OpenGL programs in such a case,
   but a program called VirtualGL is able to intercept OpenGL calls,
   make them render into a special buffer, and make the rendered result
   available to be sent to the remote computer, possibly via an
   intermediate VNC server. This process, of course, means that the data
   leaves the GPU.
