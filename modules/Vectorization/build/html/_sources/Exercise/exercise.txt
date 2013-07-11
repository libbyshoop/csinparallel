Using Vectorization
===================

Setup
-----

Download the code to be modified :download:`cdms_vector.cpp <cdms_vector.cpp>` and some data to compute with, :download:`data2.txt <data2.txt>`. This data is a table of data about particle position and motion. Columns 13, 14, and 15 represent the normalized momentum of a detected particle in the x, y, and z directions.

Modifying the Existing Code
---------------------------------

Open up "cdms_vector.cpp" and take a look at the written code. As it stands, this code simply uses the read_cols function reads in columns 13, 14, and 15 of the data, described above, and stores them in the 2-dimensional ``data`` array. The code then initializes variables for timing the computation, before timing nothing in the following lines:

.. literalinclude:: cdms_vector.cpp
    :language: c++
    :lines: 64-68

We want to add a computation that can be vectorized, so at the ``YOUR COMPUTATION HERE`` message, write a loop to calculate the total momentum of each particle and store it in a new single-dimensional array.  Since the momentums have been normalized, print statements shoould be able to verify a result of 1 for each entry.

.. .. [1] Remember that the the total momentum is `\sqrt{ p_x^2 + p_y^2 + p_z^2 }`, not just the sum.


Compiling With and Without Vectorization
----------------------------------------

Now, compile the program in two different ways, with and without vectorization. 

First, compile without the vectorization by using the -O1 flag::

    icpc -O1 -std=c99 cdms_vec.cpp -o cdms_novec

Second, compile with vectorization by allowing auto-vectorization to occur::

    icpc -std=c99 -vec-report2 cdms_vec.cpp -o cdms_vec

Now run both of those executables, noting the run times::

    % ./cdms_novec
    Total time elapsed: 0.00904894 seconds
    % ./cdms_vec
    Total time elapsed: 0.00875807 seconds

You may want to run each command multiple times to get a more representative sample.

Improving the Vectorization Performance
---------------------------------------

In order to improve the vectorization performance (or allow vectorization if the previous code did not vectorize), we can align the data as mentioned in the previous pages. When writing the data into the data array (inside the readCols function), insert ``__attribute__((aligned(16)))`` just before the semicolon at the end of the line where each column is assigned. Compile again with vectorization and note any performance improvement.

Trying Another Computation
--------------------------

Now try another computation: copy "cdms_vector.cpp" and, instead of calculating the total momentum of each particle, calculate the average x-momentum (column 13) of all the particles. Try to see what happens and whether this can be vectorized. Pragma compiler commands may or may not be useful (e.g. ``#pragma vector always``).
