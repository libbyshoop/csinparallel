***************************
Example Programs and Labs
***************************

Contents
~~~~~~~~~

* `Example Programs`_
* `Labs`_
	- `CUDA-fying the Game of Life`_
	- `The Cross-Over Point`_
* `Resources`_

Example Programs
------------------

We've written a small collection of concise and well-commented examples of CUDA programs. They are listed in order of increasing complexity. An educator could use these as a basis for lab exercises by presenting them with some parts removed and prompting the student to finish the program.

	1. addition.cu_
	#. vector_addition.cu_: A complete and slightly more complex (than the inline example on this page) vector addition program that shows how to use a 1-dimensional form of tiling to accomodate vectors that don't fit inside a single block.
	#. matrix_multiplication.cu_: This demonstrates the use of 2D grids and blocks, and tiling. It includes a reference CPU matrix-multiplication implementation along with the CUDA implementation. The use of shared memory is left out; the addition of shared memory use could be an exercise.

.. _addition.cu: http://legacy.lclark.edu/~jmache/parallel/CUDA/examples/addition.cu
.. _vector_addition.cu: http://legacy.lclark.edu/~jmache/parallel/CUDA/examples/vector_addition.cu
.. _matrix_multiplication.cu: http://legacy.lclark.edu/~jmache/parallel/CUDA/examples/matrix_multiplication.cu 

Labs
------

Try these labs after you learn more about CUDA syntax and usage in class or from some other resource. The labs are in no particular order, but the Game of Life lab is probably more challenging than the other one. Instructors should feel free to contact Chris M. for a working solution to the Game of Life lab.

CUDA-fying the Game of Life
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Practice putting all of the CUDA-pieces together by converting a serial Game of Life implementation into one that uses CUDA. You can start by using gol.c, a CPU-only implementation, as a base for writing your CUDA-enabled version. This is a good lab to apply a technique you may have learned called "tiling".

################################

**Tip**

gol.c_ uses X to draw the Game of Life board. If you can't use X or don't have the Xlib libraries, use gol_textual.c_ instead. That version prints the board to the console -- no fancy graphical display.

.. _gol.c: http://legacy.lclark.edu/~jmache/parallel/CUDA/labs/Game-of-Life/gol.c
.. _gol_textual.c: http://legacy.lclark.edu/~jmache/parallel/CUDA/labs/Game-of-Life/gol_textual.c


################################

The Cross-Over Point
~~~~~~~~~~~~~~~~~~~~~

CUDA really shines when given problems involving lots of data, but for small problems, using CUDA can be slower than a pure CPU solution. Since it can be difficult to get a feel for how large a problem needs to be before using the GPU becomes useful, this lab encourages you to find the "crossover point" for vector addition. Specifically: how large do the vectors need to be for the speed of GPU vector addition to eclipse the speed of CPU vector addition?

Modify the vector_addition.cu_ example to time how long it takes the CPU and GPU vector addition functions to operate on vectors of different magnitudes. Find (roughly) what magnitude constitutes the cross-over point for this problem on your system.

.. _vector_addition.cu: http://legacy.lclark.edu/~jmache/parallel/CUDA/examples/vector_addition.cu

################################

**Tip**

  * For a high-resolution wall clock, use the ``gettimeofday`` function by including ``sys/time.h``.
  * In some C implementations, large arrays should be allocated with ``malloc`` instead of implicitly (like ``int foo[3]``) to avoid segmentation faults.

################################

Resources
-----------

*CUDA, Supercomputing for the Masses*
	<http://drdobbs.com/high-performance-computing/207200659>

	A comprehensive twenty part tutorial series for the CUDA API that covers everything from basic memory management to details like OpenGL interoperability. These tutorials are easy to read and come with source code examples.

*Programming Massively Parallel Processors: A Hands-on Approach* by David Kirk and Wen-mei W. Hwu
	This book is a good resource on CUDA chip architecture. Programming Massively Parallel Processors is not as easy to read as the tutorial series on the Dr. Dobbs site, and does not have coverage of specific CUDA Toolkit features. The book also comes with a set of labs. In our experience, we found the labs either too easy or frustratingly hard. The difficulty came from the fact that some labs depend on information from following chapters or from trying to understand the skeleton code provided with each lab.

*CUDA By Example: An Introduction to General-Purpose GPU Programming* by Jason Sanders and Edward Kandrot

*NVIDIA CUDA C Programming Guide Version 4.0*
	<http://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_C_Programming_Guide.pdf>

	The CUDA C Programming Guide is a great resource for finding answers to specific questions. Therefore it is highly recommended to have a copy on hand as you begin your exploration of CUDA. Unfortunately, the programming guide does not provide a curriculum from which you can learn the CUDA language. The programming guide does come with many code examples which proved valuable in our learning process. The programming guide is more of a reference than a textbook.
