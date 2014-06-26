***********************************
Build and Run the Parallel Version
***********************************

When you create the executable for this program, you will need to set some flags that are particular for your machine, particularly if you want to run it with the graphical display, which uses the X11 library.  This should work on linux machines and Mac OS X machines that have X11 installed.

Lines 12-14 in the Makefile, shown below and included with the code, are where you set paths to the X11 library and include directories.  On some linux machines you may not need to set either of these, which is why they are commented out.

In this case, lines 20 and 22 are commented because rather than seeing the display, we want to start looking at how the parallel code runs (real code wouldn't use the display for simulation modeling).  When rigging the code to test for performance, you really want to eliminate most of the output, so we have just left line 24 uncommented to see the final statistics after the whole simulation is completed.

.. literalinclude:: Makefile
	:language: basemake
	:linenos: 

Build
******

::

	make

Run
****

::

	./Pandemic-openmp

The default values start with a simulation of approximately 50 people.  The OpenMP code will also use a default number of threads, which it determines from your machine's hardware configuration.

To see what elements of the computation you can change, try this:

::

	./Pandemic-openmp -?

It should give you something like this:

::

	/Pandemic-openmp -?

	./Pandemic-openmp: illegal option -- ?
	Usage: ./Pandemic-openmp [-n number_of_people][-i num_initially_infected][-w environment_width]
	[-h environment_height][-t total_number_of_days][-T duration_of_disease]
	[-c contagiousness_factor][-d infection_radius][-D deadliness_factor]
	[-m microseconds_per_day] [-p number of threads]

Note that these are defined and set in the *parse_args()* function in Initialize.h.  There is a new option, -p, for setting the number of threads.

Now you can experiment running different problem sizes with different numbers of threads, like this:

::

	time ./Pandemic-openmp -n 70000 -m 0 -p 4
	time ./Pandemic-openmp -n 70000 -m 0 -p 8

To think about
***************

There are preferable ways to instrument your code to time it, using the OpenMP function *opm_get_wtime()*.  Investigate how to use it and update this code to print running times of various sections of the code.  What loop takes the most time?

Can you calculate the speedup you get by using varying numbers of threads for a fairly large problem size?
