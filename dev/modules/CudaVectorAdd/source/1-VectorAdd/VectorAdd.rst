*******************
Vector Addition Lab
*******************

.. warning:: The Unified Memory parts of this lab may not work on your machine.
    Run the devics program found in the last section to find out if 
    your machine is compatable

Research Questions
##################

- For what size prolem is the CUDA computation faster than the sequential 
  computation?

- What effect does using Unified Memory have on the speed of our program?

Getting started
###############

Download and extract vectorAdd.zip into a directory named vectorAdd.
Open vectorAdd.cu and vectorAdd6.cu and familiarize yourself with the code.
Compile and run the programs.

.. note:: skip compiling vectorAdd6.cu if your machine is incompatable

Run the programs and see what happens.

Exercise
########

Part One: cudaMemcpy
********************

Using omp_get_wtime() modify vectorAdd.cu so that it reports

#. The time required by the CUDA computation specifically

   a. The time spent allocating A, B, and C
   #. The time spent copying A and B from the Host to the device
   #. The time spent computing the sum of A and B into C
   #. The time spent copying C from the device to the host
   #. The total time of the CUDA computation (i.e., the sum of a-d)

#. The time required by the sequential computation

None of these times should include any I/O so make sure you comment out
the printf() statements.

Use the Makefile to build your modified version of the program. 
When it compiles successfully run it as follows: 

``./vectorAdd`` 

The program's default array size is 50,000 elements

In a spreadsheet record and label your times in a column labeld 50,000. Which
is faster, the CUDA version or the CPU version?

Repeat this problem with a larger array. Run it again with 500,000 elements.

``./vectorAdd 500000``

Record your results. Repeat the process again wih 5,000,000 elements, 
50,000,000 and 500,000,000 elements. How do these times compare?
Were you able to run all of them succesfully? If not why?

Create a line chart, with one line for the sequential code's 
times and one line for the CUDA code's total times. 
Your X-axis should be labeled with 50,000 500,000 
5,000,000 and 50,000,000 your Y-axis should be the time.

Then create a "stacked" barchart of the CUDA times with the same X and Y axes
as your first chart.. For each X-axis value, this chart should "stack" 
the CUDA computation's

#. allocation time
#. host-to-device transfer time
#. computation time
#. device-to-host transfer time

What observations can you make about the CUDA vs the sequential computations? 
How much time does the CUDA computation spend transferring data compared to computing? 
What is the answer to our first research question?

Part Two: Unified Memory
************************

.. note:: skip this section if your device is not compatable with Unified Memory.

Using omp_get_wtime() modify vectorAdd6.cu so that it reports

#. The time required by the CUDA computation specifically
   
   a. The time spent allocating A, B, and C
   #. The time spent computing the sum of A and B into C
   #. The total time of the CUDA computation (i.e., the sum of a and b)

#. The time required by the sequential computation

Again, none of these times should include any I/O so make sure you comment out
the printf() statements.

Run your program using

``./vectorAdd6`` 

Record your results using 50,000 500,000 5,000,000 and  
50,000,000 elements. How do these times compare?

Add this new data to the line chart and stacked bar charts from part one.
How does using unified memory compare to using ``cudaMemcpy``\ ?
What is the bottleneck for the ``cudaMemcpy`` version? What about
the unified memory version?
