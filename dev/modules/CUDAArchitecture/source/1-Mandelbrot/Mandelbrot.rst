Mandelbrot Test Code
====================

Choosing a good number of blocks and threads per block is
an important part of CUDA Programming. To illustrate this, we
will take a look at a program that generates images of the
Mandelbrot set. To run the programs you will need a CUDA capable
machine running Mac or Linux as well as the appropriate XOrg
developer package. Download :download:`mandelbrot.cu` and the 
:download:`Makefile` and run ``Make all`` This will generate 3
programs. 

Mandelbrot is a mandelbrot set viewer designed for demonstrations
. It allows you to zoom in and out and move around the 
Mandelbrot set. The controls are w for up, s for down, a for
left, d for right, e to zoom in, q to zoom out and x to exit.

benchmark runs the computation without displaying anything and
prints out the time it took before exiting.

XBenchmark is a hybrid that prints out the computation time and
allows you to move around. This is useful because the computation
time is dependent on your position within the Mandelbrot set.

Each of the programs takes betwen 0 and 4 commandline arguments

#. the number of blocks used by the kernel
#. the number of threads per block
#. the image size in pixels, the image is always square
#. the image depth (explained later)

What is the Mandelbrot set?
###########################

The mandelbrot set is defined as the set of all complex numbers C
such that the formula Z\ :sub:`n+1` = Z\ :sub:`n`\ :sup:`2` + C 
tends towards infinity. If we plot the real values of C on the X
axis and the imaginary values of C on the Y axis we get a two 
dimensional fractal shape.

If that doesn't help just think of the strange but pretty shape 
that's on the cover of every math textbook published since 1980.

Coding the Mandelbrot set
#########################

The to determine whether a value is in or out of the mandelbrot
set we loop through the formula  Z\ :sub:`n+1` = Z\ :sub:`n`\
:sup:`2` + C a certain number of times (this is the image depth
from earlier) and during each iteration, check if the magnitude 
of Z is greater than 2 we return false. However we want our
Mandelbrot image to look pretty, so instead we'll return the
iteration in which it went out of bounds and and then interpret
that number as a color. If it completes the loop without going
out of bounds we'll assign it the color black

After some algebraic manipulation to reduce the number of
floating point multiplications, our code looks like this:

.. code-block:: cu
    
    __device__ uint32_t mandel_double(double cr, double ci, int max_iter) {
        double zr = 0;
        double zi = 0;
        double zrsqr = 0;
        double zisqr = 0;
        
        uint32_t i;
        
        for (i = 0; i < max_iter; i++){
            zi = zr * zi;
            zi += zi;
            zi += ci;
            zr = zrsqr - zisqr + cr;
            zrsqr = zr * zr;
            zisqr = zi * zi;
            
            //the fewer iterations it takes to diverge, the farther from the set
            if (zrsqr + zisqr > 4.0) break;
        }
        return i;
    }

But wait didn't we say in the last module that conditionals 
don't speed up CUDA code? Yes, usually they don't however
due to the nature of the mandelbrot set it is very likely
that some warps have threads that all terminate before 
reaching ``max_iter`` so in some cases it will give us a
slight speed up. If the warp contains a point within the 
Mandelbrot set, we won't get any speed up from breaking.

We also need a kernel that will divide the pixels between
the threads and run ``mandel_double`` on each of them
Our code is as follows where ``dim`` is the image dimension,
``counts`` is the list representing our image, and ``step``
represents the distance between the points represented by 
the pixels:

.. code-block:: cu

   __global__ void mandel_kernel(uint32_t *counts, double xmin, double ymin,
            double step, int max_iter, int dim, uint32_t *colors){
        int pix_per_thread = dim * dim / (gridDim.x * blockDim.x);
        int tId = blockDim.x * blockIdx.x + threadIdx.x;
        int offset = pix_per_thread * tId;
        for (int i = offset; i < offset + pix_per_thread; i++){
            int x = i % dim;
            int y = i / dim;
            double cr = xmin + x * step;
            double ci = ymin + y * step;
            counts[y * dim + x]  = colors[mandel_double(cr, ci, max_iter)];
        }
        if (gridDim.x * blockDim.x * pix_per_thread < dim * dim
                && tId < (dim * dim) - (blockDim.x * gridDim.x)){
            int i = blockDim.x * gridDim.x * pix_per_thread + tId;
            int x = i % dim;
            int y = i / dim;
            double cr = xmin + x * step;
            double ci = ymin + y * step;
            counts[y * dim + x]  = colors[mandel_double(cr, ci, max_iter)];
        }
        
    }
 
In order to compensate for block and grid dimensions that
do not easily divide the picture we make the first threads
pick up the 'slack.' This is also the reason why we are not
using 2 dimensional grids and blocks.

That's the meat of the program, feel free to explore the
it on your own, most of the rest of the program is dedicated
to displaying the data generated by these 2 functions.
