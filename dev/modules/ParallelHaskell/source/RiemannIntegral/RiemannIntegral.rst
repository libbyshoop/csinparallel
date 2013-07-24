Riemann Integral and Performance
===================================

For demonstrating some of the features of Haskell, an instructor could show or assign an implementation of a
simple program that will estimate π by approximating the area under half of the unit circle and multiplying
it by two. This has been a popular introduction to parallelism and has been used in the ACM Parallel
Computing TechPack \ :sup:`[11]`\. There are multiple parallel implementations possible in Haskell, but performance
optimization proves to be very difficult for some of them. We first implemented a π estimation using the
``par`` and ``pseq`` functions. Optimization proved especially difficult with this implementation because using a recursive par function creates a new thread for every Riemann rectangle calculated, and the resulting overhead outweighed the performance gains from the parallel execution. In the end, this code was not able to outperform the sequential version of the program.

A slightly more abstracted tool, ``parMap``, provides slightly better results for this program. The function
is a combination of the standard map function from ``Data.List``, in combination with a ``parList`` function
from ``Control.Parallel``. The code converts a partition list into a list of areas of Riemann rectangles, by
partially applying an area function to each element in the list in parallel. While ``parMap`` is a clean way to
calculate the areas of each rectangle in parallel, it fails to solve the problem already faced by the ``par`` and
``pseq`` implementation: excessive thread creation does not improve the program performance on a computer
with only a few CPUs. If we try to compute the areas of 20 million Riemann rectangles on a computer
that can only execute a few dozen threads at the same time, then our code accomplishes very little beyond
overloading the job scheduler. Both ``parMap`` and the ``par`` and ``pseq`` implementations may have more potential
on massively parallel systems, perhaps with access to a GPU, but their very nature slows them down on less
ambitious machines. Again, we were unable to create an implementation with ``parMap`` that could run faster
than the sequential version.

These programs resulted in discouraging performances, but we did finally utilize a parallel tool that resulted in competitive run times. The ``parListChunk`` function is a special abstraction in the ``Control.Parallel`` module that allows the programmer to apply a map to a list by dividing the work into parallel "chunks" of data of a specified size \ :sup:`[13]`\. This implementation allowed for more specific performance tuning based on the number of cores the code was being run on. It was an efficient way to divide out the workload, and proved to be scalable by simply adding an argument at compile time \ :sup:`[13]`\. This meant that we were able to maximize the amount of work done by each core by sending each CPU set chunks of our partition list in order to calculate areas. Since the chunks were divided up beforehand, inter-core communication was kept to a minimum until the reduction step. Here is the source code for our parallel solution, using ``parListChunk``. This implementation achieved a significant speedup when compared to our sequential program, which can
be seen in our results section. ::

	1   --Creates a list of Riemann Rectangle areas
	2   --given a partition set and a delta x
	3   parEstimation2 delt xs chunks = map (area delt) xs
	4     'using' parListChunk (floor ((2.0 / delt) / chunks)
	5   --Left-folds the set of Riemann rectangle areas
	6   --together into a sum. Multiplies by 2 to get pi
	7   parPiEstimate2 n chunks =
	8     2 * (foldl (+) 0 (parEstimation2 (2 / n)
	9     [(-1), (-1 + 2 / n) .. 1] chunks))
	10   --Calculates the area of a Riemann rectangle
	11   area y x = y * (circle x)
	12   --The equation for the upper hemisphere of the unit circle
	13   circle :: Double -> Double
	14   circle x = sqrt (abs(1 - x^2))

When using this simple program as an educational module in an undergraduate classroom, we think that
all three of these implementation can prove valuable. The ``parListChunk`` implementation yields the most
compelling results, but the ``parMap``, ``par`` and ``pseq`` tools provide valuable insight into the nature of parallel
programming in Haskell. Furthermore, their failure to outperform the sequential implementation will give
students insight into the hardware limitations that are often faced when attempting to parallelize code.
We have posted on our website_, and included as an appendix, a challenging but introductory assignment
that is meant to encourage discussion from students about the benefits and pitfalls in using the ``par`` and
``pseq`` functions in order to parallelize a simple quicksort. Keep in mind that such a parallel implementation
does not actually produce an increased performance without some very careful tuning, but students should
recognize why this is the case. Creation of a massive amount of threads for a small operation, all of which
must communicate with each other, results in overhead that can outweigh the benefits of parallel evaluation.
Students should start with the rough, inefficient parallel code. The instructor can then guide the class
through the process of optimization, which will provide valuable insight into the inner workings of the
parallel tools.

Performance
---------------
We tested our Riemann program on a 12 core computer. We found that the parallel program improved in
performance across 1, 2, 4, 8, and 12 cores. The run time for estimating 20 million partitions are in the
table. The program was executed with the following command::

	./Riemann +RTS -Nfnumber of coresg -H8000m -K8000m -qb -RTS 20000000 {number of chunks}

This command allocates 8,000 MB and 8,000 MB of memory for the stack and heap, respectively. As
mentioned before, we also used the ``-qb`` option to disable parallel garbage collection. Our program also took
two command line arguments. The first argument specifies the number of partitions to be calculated, and
the second specifies the number of chunks. In our tests, we told the program to split into as many chunks
as we had cores. The performance levels off at 8 cores. The sequential code was also run on 1 core with an
average run time of 19.40 seconds. The average parallel Riemann run time ordered by number of cores is shown in the table below.

.. image:: CAPTURE.png
	:height: 100px
	:align: center

.. _website: http://legacy.lclark.edu/~jmache/parallel/haskell