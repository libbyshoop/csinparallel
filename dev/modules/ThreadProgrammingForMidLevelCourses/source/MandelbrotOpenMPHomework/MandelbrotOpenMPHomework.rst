**************************
Mandelbrot OpenMP Homework
**************************

#. Complete four parallel versions of the mandelbrot program. These versions should do the following (all descriptions focusing on the nested pair of loops that set the pixels):

   a) Parallelize the outer loop with the loops in the order they appear in the given code. (So the pragma is associated with the ``i`` loop.)

   b) Parallelize the outer loop after swapping the loop order. (Moving the ``j`` loop to the outside and associating the pragma with it).

   c) Parallelize the inner loop with the loops in the order they appear in the given code. (Associating the pragma with the ``j`` loop, but without swapping the loop order.)

   d) Parallelize the inner loop after swapping the loop order.

Calculate running times for the serial code and each of the parallel versions. For calculating these running times, be sure to run the ``setenv`` command from the lab so that you run with 2 threads. Also calculate the speedup achieved by each version.

#. Is the overhead of parallelizing this program increased or decreased by putting the pragma on the inner loop (c and d above) as opposed to the outer loop (a and b)? Be sure to explain your answer. 

#. Does adding ``schedule(dynamic)`` to the end of the pragma increase or decrease the overhead of parallelizing this program? Is the load balance improved or worsened by doing so? Be sure to explain both answers. 

#. Consider 3 parameters in the code: ``numRows``, ``numCols``, and ``maxIteration``. The first two determine the image size and the third is the number of iterations at which the mandelbrot function stops. How would changing each of these affect (i.e. increase or decrease) the percentage of the program that runs serially? How would doing so change the speedup achieved by your parallel versions? Be sure to explain both answers. 

