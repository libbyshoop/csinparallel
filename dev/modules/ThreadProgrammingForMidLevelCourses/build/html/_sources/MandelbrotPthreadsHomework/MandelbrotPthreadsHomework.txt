****************************
Mandelbrot pthreads Homework
****************************

#. Complete two parallel versions of the mandelbrot program. The first should parallelize the main nested loop in the order it appears in the given code (splitting iterations of ``i`` between 2 threads). The second should do so after swapping the loop order (moving the ``j`` loop to the outside and splitting its iterations between 2 threads). Calculate running times for the serial code and both of the parallel versions. Also calculate the speedup achieved by each version.

#. Does the ``mandelbrot`` function take longer for a pixel that it determines to be in the Mandelbrot set or out of it? (Recall that a point in the set appears black in the diagram and is one for which the ``mandelbrot`` function returns 0.) Explain why.

#. Based on your answer to the previous question and looking at the diagram of the Mandelbrot set, does parallelizing the loops as written or swapping them first give a better load balance? Explain why.

#. Consider 3 parameters in the code: ``numRows``, ``numCols``, and ``maxIteration``. The first two determine the image size and the third is the number of iterations at which the ``mandelbrot`` function stops. How would changing each of these affect (i.e. increase or decrease) the percentage of the program that runs serially? How would doing so change the speedup achieved by your parallel versions? Be sure to explain both answers. 
