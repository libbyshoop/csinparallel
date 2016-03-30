***********************************
Build and Run the Parallel Version
***********************************

When you create the executable for this program, you will need to set some flags that are particular for your machine, particularly if you want to run it with the graphical display, which uses the X11 library.  This should work on linux machines and Mac OS X machines that have X11 installed.

Lines 13-15 in the Makefile, shown below and included with the code, are where you set paths to the X11 library and include directories.  On some linux machines you may not need to set either of these, which is why they are commented out.

In this case, lines 13 and 15 are commented because rather than seeing the display, we want to start looking at how the parallel code runs (real code wouldn't use the display for simulation modeling).  When rigging the code to test for performance, you really want to eliminate most of the output, so we have just left line 15 uncommented to see the final statistics after the whole simulation is completed.

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

	mpirun -machinefile machines -np 6 ./Pandemic-mpi

Your instructor will provide a machines file for your cluster. You can eliminate the use of the
-machinefile machines option if you are running multiple processes on the same machine.

The default values start with a simulation of approximately 50 people. 

To see what elements of the computation you can change, try this:

::

	./Pandemic-mpi -?

It should give you something like this:

::

	/Pandemic-mpi -?

	Usage: ./Pandemic-mpi [-n total_number_of_people][-i total_num_initially_infected][-w environment_width][-h environment_height][-t total_number_of_days][-T duration_of_disease][-c contagiousness_factor][-d infection_radius][-D deadliness_factor][-m microseconds_per_day]

Note that these are defined and set in the *parse_args()* function in Initialize.h.  

Now you can experiment running different problem sizes with different numbers of threads, like this:

::

	 mpirun -machinefile machines -np 6 ./Pandemic-mpi -n 70000 -m 0 
	 mpirun -machinefile machines -np 8 ./Pandemic-mpi -n 70000 -m 0 

To think about
***************

There are preferable ways to instrument your code to time it, using the MPI function *MPI_Wtime()*.  Investigate how to use it and update this code to print running times of various sections of the code.  What loop takes the most time?

Can you calculate the speedup you get by using varying numbers of processes for a fairly large problem size?
