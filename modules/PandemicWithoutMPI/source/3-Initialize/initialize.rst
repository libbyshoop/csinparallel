********************
Initialize Functions
********************

init
****

This function will first initialize variables in the constant structure with default values. It will also initialize **number_of_people** variable and **num_initially_infected** variable. After this, it will set all the counters in side stats structure to zero, as well as state counters inside global struct.

Then, **Init()** function will call the following 5 functions to finish the initialization process.

parse_args
**********

.. figure:: img-6.png
   :align: center
   :alt: image

These parameters are specified via command line arguments when the
program is run. Otherwise, default values are used. The code uses **getopt**
function to do this. Type **man 3 getopt** in a shell if you are interested
how it works.

init_check
**********

This function makes sure that the **total number of initially infected people** is less than the **total number of people**

.. figure:: img-7.png
   :align: center
   :alt: image

The simulation can’t run if there are more initially infected people
than there are people. If there are, the code uses the fprintf function
to print an error message to standard error, and it exits the program
with exit code -1.

allocate_array
**************

At this point we are ready to allocate our arrays, which must be performed before we can start filling the arrays. Allocating an array means reserving enough space in memory for it; if we don’t reserve the space the program will assume that it is a zero-length array. The allocation must happen in the “heap” memory, meaning we must allocate it dynamically (i.e. as the program is running). To allocate memory on the heap, we use the **malloc** function, which takes the amount of space that is requested and returns a pointer to the newly allocated memory, which we can then use as an array. Let’s see an example with the **x_locations** array:

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 177

Here we see that malloc has taken an argument, **number_of_people
* sizeof(int)**. This is how we specify that we want to fill the array
with a certain number of integers, namely the amount stored in the
**number_of_people** variable. We also need to specify how big
these integers are, for which we use the **sizeof(int)** function. We then
take the return from **malloc** and tell the program to “cast” it (i.e. use
it) as a pointer to integers, for which we use **(int*)**. This is then
assigned to **x_locations**, and we can now use **x_locations** as an array.

For the 2D array **environment**, we must allocate not only the array itself
but also each of the arrays that it contains (since a 2D array is an
array whose elements are arrays). The array has horizontal strips of
length **environment_width** and vertical strips of length
**environment_height**. We perform the allocation by allocating enough
space for the entire array first using 

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 186-187

That is, we are allocating enough **char\***’s for
**environment_width** times **environment_height**, casting this as a **char\*\***
and assigning it to **environment**. Then we allocate each array within
**environment**, like so:

.. literalinclude:: Initialize.h	
    :language: c
    :lines: 186-195

The number of arrays we need is stored in **environment_width**, so we loop
from **0** to **environment_width – 1** and allocate enough space in each
element of environment for **environment_height** chars, each one of which
has size **sizeof(char)**.

This can be a convoluted process but is necessary for allocating arrays
dynamically, which allows us to specify options on the command line (so
we don’t have to edit the source code and re-compile each time we want
to run a simulation with different parameters).

init_array
**********

First, the function set the states of the initially infected people and set the count of its infected people

.. figure:: img-12.png
   :align: center
   :alt: image

Threads set the states of infected people using the **states** array.
They fill the first **num\_initially\_infected** cells of the array
with the **INFECTED** constant; i.e. they fill in the **0** through
**num\_initially\_infected – 1** positions of the array with **INFECTED**
as below:

.. literalinclude:: Initialize.h
    :language: c
    :lines: 212-219

Note we also add 1 to the **num\_infected variable** using the plus-plus operator (++) at each iteration of the loop. This is how we count the number of infected people.

Next, the function set the states of the rest of its people and set the count of its susceptible people

.. figure:: img-13.png
   :align: center
   :alt: image

This is similar to last step, but we want to fill the rest of the array
(from **num\_initially\_infected** to **number\_of\_people - 1**) with
the **SUSCEPTIBLE** constant, and we want to add **1** to the
**num\_susceptible** variable at each iteration of the loop:

.. literalinclude:: Initialize.h
    :language: c
    :lines: 221-229

The **states** array is now full; the first **num\_initially\_infected** cells have the **INFECTED** constant, and the rest have the **SUSCEPTIBLE** constant.

Then, the function sets random x and y locations for each people

.. figure:: img-14.png
   :align: center
   :alt: image

Locations of people are stored in the **x\_locations** and **y\_locations** arrays. To fill these arrays with random values, we use a for loop and the random function:

.. literalinclude:: Initialize.h  	
    :language: c
    :lines: 231-239

By calling random with the **modulus (%)** operator, we can restrict the
size of the random number it generates. Since we cannot have x locations
larger than the width of the environment, we call **random() %
environment\_width**; to make sure the **x location** of each person is less
than **environment\_width**. Similarly for the **y location** and
**environment\_height**.

We are filling the x and y location arrays for all of the people for
which the process is responsible, so we loop from **0** to **number\_of\_people – 1**.

Finally, the function initializes the number of days infected of each of its people to 0

.. figure:: img-15.png
   :align: center
   :alt: image

The number of days each person is infected is stored in the **num\_days\_infected** array, so we loop over all of the people and
fill this array with 0, since the simulation starts at day 0, at which
point no days have yet elapsed:

.. literalinclude:: Initialize.h	
    :language: c
    :lines: 241-248