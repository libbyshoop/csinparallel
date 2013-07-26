***********************************************
Finding the Area Under the Curve Using Blocking
***********************************************

Once each process has the number of rectangles it is responsible for it can start looping through this number, calculate and store the area one by one. If this loop is parallelized using OpenMP by default OpenMP will split the data in half amongst the threads. By each process or thread having its own set of rectangles to loop through, we have employed the concept of blocking as a way of parallel computation. The following code excerpt shows the loop:

.. literalinclude:: area.h
	:language: c
	:lines: 141-178

The next image shows how the blocking is done using OpenMP only on the head node. Each thread has its unique color.

.. image:: openmp.png
	:width: 350px
	:align: center
	:height: 700px
	:alt: openmp

The source code can be downloaded :download:`here <area.tgz>`.