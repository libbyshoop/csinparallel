***********************************
Build and Run
***********************************

When you create the executable for this program, you will need to set some flags that are particular for your machine, particularly if you want to run it with the graphical display, which uses the X11 library.  This should work on linux machines and Mac OS X machines that have X11 installed.

Lines 12-14 in the Makefile, shown below and included with the code, are where you set paths to the X11 library and include directories.  On some linux machines you may not need to set either of these, which is why they are commented out.

If you want to use a text display instead of the graphical X11 display, uncomment line 20 and comment line 22.  When rigging the code to test for performance, you might want to eliminate the display of each iteration completely, in which case you can comment line 20 and 22.

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

	./Pandemic-serial

The default values start with a simulation of approximately 50 people.

To see what elements of the computation you can change, try this:

::

	./Pandemic-serial -?

It should give you something like this:

::

	./Pandemic-serial: illegal option -- ?

	Usage: ./Pandemic-serial [-n number_of_people][-i num_initially_infected 
	[-w environment_width][-h environment_height][-t total_number_of_days]
	[-T duration_of_disease][-c contagiousness_factor][-d infection_radius]
	[-D deadliness_factor][-m microseconds_per_day]

Note that these are defined and set in the *parse_args()* function in Initialize.h.

If you comment out lines 20 and 22 of the Makefile shown above, you can try getting some preliminary basic sense of how fast the program runs with various sizes of the problem (in this case the number of people).  First, comment lines 20 and 22 in the Makefile and do the following:

::

	make clean
	make

Then execute various problem sizes, taking no time between iterations, as follows:

::

	time ./Pandemic-serial -n 20000 -m 0

Experiment by changing the value of n higher and lower.  You should see the program take more time as you increase the number of people.  Experiment with some of the other parameters also. Do some investigation of what the unix time command does.  This is a very rough way to see how fast your program runs.  There are preferable ways to instrument the code itself that you could investigate.

