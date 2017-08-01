**************************************
Trapezoidal Rule Integration
**************************************


Trapezoidal Rule
=====================
The trapezoidal rule is a technique for approximating the region under a
function, :math:`y = f(x)`, using trapezoids to calculate area. The process is quite
simple. Let a and b represent the left and right endpoints of the function.
The interval [a,b] is divided into subintervals. For each subinterval,
the function is approximated with a straight line between the function
values at both ends of the subinterval. Each subinterval is now a trapezoid.
Lastly, the area of each trapezoid is calculated and all areas are
summed to get an approximation of the area under the function.

Parallelization
=====================
In order to parallelize this rule, we must identify the necessary tasks and
decide how to map the tasks to the processes. Tasks include finding the area of
many single trapezoids and then summing these areas together. Intuitively,
as we increase the number of trapezoids, we will receive a more accurate
prediction for the area under the curve. Thus, we will be using more
trapezoids than cores in this problem and we will need to split up the computations
for calculating the areas of the trapezoids. We choose to do this by assigning
each process a subinterval that contains the number of trapezoids obtained
from the calculation of the total number of trapezoids divided by number of processes.
This assumes that the total number of trapezoids is evenly divisible by the number
of processes. Each process will apply the trapezoidal rule to its subinterval.
Lastly, the master process adds together the estimates.

.. image:: TrapComputeArea.png
	:width: 600

Code
================
*file: MPI_examples/trapIntegration/mpi_trap/mpi_trap.c*

The code for this example is from Peter Pacheco's book, An Introduction to Parallel Programming.
For further implementations and reading corresponding to this example, refer
to his book which is listed below.

* **Book:** *An Introduction to Parallel Programming*, Peter Pacheco, Morgan Kaufmann Publishers, 2011

*Build inside mpi_trap directory:*
::

  make mpi_trap

*Execute on the command line inside mpi_trap directory:*
::

  mpirun -np <number of processes> ./mpi_trap

.. literalinclude:: ../MPI_examples/trapIntegration/mpi_trap/mpi_trap.c
    :language: c
    :linenos:

Global and Local Variables
============================
In MPI, local variables only are important only to the process using them.
Local variables in this problem include *local_a*, *local_b*, and *local_n*.
Note that the values of *local_a* and *local_b* are completely dependent
upon process rank. They must be specifically calculated for each process
to ensure that each process receives a different subinterval. The variable
*local_n* remains the same for every process.

In contrast, variables that are important to all processes are global
variables. Variables *a*, *b* and *n* are some global variables in this
example. These variables do not change values during the duration of the program.

Trap Function
====================
This function implements the trapezoidal rule for the interval given as input.

To calculate the area of a single trapezoid, we need to know the left and right
endpoints, and the length of the trapezoid.  Let *a*, *b* and *h* represent the left
endpoint, right endpoint and length respectively. The function values at these
endpoints are *f(a)* and *f(b)*. The area of the trapezoid is as follows:

**Area of one trapezoid** =

.. math::
    \frac{h}{2} [f(a) + f(b)]

However, in our problem there are many subintervals and each subinterval may
contain multiple trapezoids. Now we have *a* and *b* representing the left
and right endpoint of function. The *n* trapezoids are of equal length, *h* where
*h* = :math:`\frac{b-a}{n}`. Let's focus on a single subinterval whose left endpoint is
*local_a* and right endpoint is *local_b*. Then the trapezoids within the interval
have the following endpoints:


[local_a, local_a + h], [local_a + h, local_a + 2h], ... , [local_a, local_a + (n-1)h, b]

The sum of the areas of the trapezoids (estimate of area of the subinterval) is:

**Subinterval area** =

.. math::

  \frac{h}{2}[f(local \textunderscore a) + f(local \textunderscore a + h) * 2 +
	f(local \textunderscore  a + 2h) * 2 + ... + f(local \textunderscore  a + (n-1)h)*2
	+ f(local \textunderscore  b)] =

.. math::

 \frac{h}{2}[f(local \textunderscore a)/2 + f(local \textunderscore a + h) +
 f(local \textunderscore a + 2h) + ... + f(local \textunderscore a + (n-1)h) +
 f(local \textunderscore b)/2]

The Trap function follows this logic closely. The function takes both left and
right endpoints, number of trapezoids within the subinterval and trapezoid
length. A for loop is used to loop through the endpoints of all of the trapezoids
within the subinterval. The function value at each of these points is accumulated
to :math:`\frac{f(left \textunderscore endpt) + f(right \textunderscore endpt)}{2}`.
Lastly, this sum is multiplied by trapezoid length to get the total area of the subinterval.

.. literalinclude:: ../MPI_examples/trapIntegration/mpi_trap/mpi_trap.c
    :language: c
    :lines: 131-143
