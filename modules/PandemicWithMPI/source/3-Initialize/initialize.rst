********************
Initialize Functions
********************

init()
******

This function will first initialize variables in the constant structure with default values. It will also initialize **total_number_of_people** variable, **total_num_initially_infected** variable and **total_num_infected** variable. After this, it will set all the counters inside stats structure to zero, as well as state counters inside global struct.

Before executing the algorithm, the code starts by initializing MPI using 

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 85-86

We pass the addresses of the arguments to **main**, **argc** and **argv**, so that MPI can strip out anything from the command line related to MPI, such as **mpirun** or **–np**. **MPI_Init** must be called before any other MPI functions are executed, and we also want to call it before we parse the rest of the command line arguments in **parse_args()** function.

.. figure:: img-5.png
   :align: center
   :alt: image

Here we see one process figuring out its rank. It does so by calling 

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 90-91

function. This function checks the MPI “world” (the “communicator” of all the MPI processes, MPI_COMM_WORLD). You pass the address of the variable for the process’s rank to the function as the second argument using the ampersand (&).

If we only have 1 process total (i.e., if we are not using distributed memory), then the rank of the process will be 0, which we set in the code as **our_rank = 0**.

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 93-94

We also see another process figuring out how many processes there are. It does so by calling 

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 90, 92

function. Just as with **MPI_Comm_rank**, you pass the communicator of all the processes and the address of the variable for the number of processes.

If we have only 1 process total, we set the number of processes by calling total_number_of_processes = 1.

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 93, 95

After MPI initialization, **init()** function will call the following five functions.

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 98, 100, 102, 108, 110-113

parse_args()
************

.. figure:: img-6.png
   :align: center
   :alt: image

These parameters are specified via command line arguments when the program is run. Otherwise, default values are used. The code uses **getopt** function to do this. Type **man 3 getopt** in a shell if you are interested how it works.

init_check()
************

This function makes sure that for each process, the total number of initially infected people is less than the total number of people

.. figure:: img-7.png
   :align: center
   :alt: image

The simulation can’t run if there are more initially infected people than there are people. If there are, the code uses the fprintf function to print an error message to standard error, and it exits the program with exit code -1.

find_size()
***********

For each process, this function determines the number of people for which it is responsible

.. figure:: img-8.png
   :align: center
   :alt: image

Each process will try to take an even split of the number of people. It does so by dividing the number of people by the total number of processes and throwing away any remainder. Because the variables involved are integers in C, the throwing away of the remainder is handled automatically in the division

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 257

The last process is responsible for the remainder

.. figure:: img-9.png
   :align: center
   :alt: image

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 259-263

Every person has to be accounted for, so any remainder of the division is assigned to the last process. We can obtain the remainder by using the modulo operator (%), and we add it to the existing value using the plus-equals operator (+=): 

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 262

We only want the last process to do this, so we surround the code with 

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 260

since the last process has rank **N–1**, where N is the total number of processes.

Each process determines the number of initially infected people
for which it is responsible

.. figure:: img-10.png
   :align: center
   :alt: image

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 267-268

This is the same method used before, but it considers only the infected people.

The last process is responsible for the remainder

.. figure:: img-11.png
   :align: center
   :alt: image

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 273-274

This is the same method used before, but it considers only the infected people.

allocate_array
**************

At this point we are ready to allocate our arrays, which must be performed before we can start filling the arrays. Allocating an array means reserving enough space in memory for it; if we don’t reserve the space the program will assume that it is a zero-length array. The allocation must happen in the “heap” memory, meaning we must allocate it dynamically (i.e. as the program is running). To allocate memory on the heap, we use the **malloc** function, which takes the amount of space that is requested and returns a pointer to the newly allocated memory, which we can then use as an array. Let’s see an example with the x_locations array:

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 214

Here we see that malloc has taken an argument, **total_number_of_people * sizeof(int)**. This is how we specify that we want to fill the array with a certain number of integers, namely the amount stored in the **total_number_of_people** variable. We also need to specify how big these integers are, for which we use the **sizeof(int)** function. We then take the return from **malloc** and tell the program to “cast” it (i.e. use it) as a pointer to integers, for which we use **(int*)**. This is then assigned to x_locations, and we can now use **x_locations** as an array.

For the 2D array **environment**, we must allocate not only the array itself but also each of the arrays that it contains (since a 2D array is an array whose elements are arrays). The array has horizontal strips of length **environment_width** and vertical strips of length **environment_height**. We perform the allocation by allocating enough space for the entire array first using 

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 230-231

That is, we are allocating enough **char\***’s for **environment_width** times **environment_height**, casting this as a **char\*\*** and assigning it to **environment**. Then we allocate each array within **environment**, like so:

.. literalinclude:: Initialize.h	
    :language: c
    :lines: 233-239

The number of arrays we need is stored in **environment_width**, so we loop from **0** to **environment_width – 1** and allocate enough space in each element of environment for **environment_height** chars, each one of which has size **sizeof(char)**.

This can be a convoluted process but is necessary for allocating arrays dynamically, which allows us to specify options on the command line (so we don’t have to edit the source code and re-compile each time we want to run a simulation with different parameters).

init_array()
************

This function can be divided into four parts.

The function sets the states of the initially infected people and set the count of its infected people

.. figure:: img-12.png
   :align: center
   :alt: image

The function also sets the states of infected people using the **our\_states** array. They fill the first **our\_num\_initially\_infected** cells of the array with the **INFECTED** constant; i.e. they fill in the **0** through **our\_num\_initially\_infected – 1** positions of the array with **INFECTED** as below:

.. literalinclude:: Initialize.h	
    :language: c
    :lines: 291-298

The function sets the states of the rest of its
people and set the count of its susceptible people

.. figure:: img-13.png
   :align: center
   :alt: image

This is similar to last step, but we want to fill the rest of the array (from **our\_num\_initially\_infected** to **our\_number\_of\_people - 1**) with the **SUSCEPTIBLE** constant, and we want to add **1** to the **our\_num\_susceptible** variable at each iteration of the loop:

.. literalinclude:: Initialize.h	
    :language: c
    :lines: 300-308

The **our\_states** array is now full; the first **our\_num\_initially\_infected** cells have the **INFECTED** constant, and the rest have the **SUSCEPTIBLE** constant.

The third step is that the function sets random x and y locations for each of its people

.. figure:: img-14.png
   :align: center
   :alt: image

Locations of people are stored in the **our\_x\_locations** and **our\_y\_locations** arrays. To fill these arrays with random values, we use a for loop and the random function:

.. literalinclude:: Initialize.h	
    :language: c
    :lines: 310-318

By calling random with the **modulus (%)** operator, we can restrict the size of the random number it generates. Since we cannot have x locations larger than the width of the environment, we call **random() % environment\_width**; to make sure the **x location** of each person is less than **environment\_width**. Similarly for the **y location** and **environment\_height**.

We are filling the x and y location arrays for all of the people for which the process is responsible, so we loop from **0** to **our\_number\_of\_people – 1**.

Finally, the function initializes the number of days infected of each of its people to 0

.. figure:: img-15.png
   :align: center
   :alt: image

The number of days each person is infected is stored in the **our\_num\_days\_infected** array, so we loop over all of the people and fill this array with 0, since the simulation starts at day 0, at which point no days have yet elapsed:

.. literalinclude:: Initialize.h	
    :language: c
    :lines: 320-327