*****************
Including OpenMP
*****************

:download:`Download Pandemic-MPI-OMP.zip <Pandemic-MPI-OMP.zip>`

It is really easy to include OpenMP features into existing code we have. All we need to do is to identify all the functions that could use OpenMP. There are in total 5 functions that could use OpenMP to increase performance. The first function is the **init_array()** function in *Initialize.h* file. The next four functions are all the core functions inside *Core.h* file.

In Initialize.h
###############

init_array()
************

This function can be divided into four parts: the first part sets the states of the initially infected people and sets the count of infected people. The second part sets states of the rest of the people and sets the of susceptible people. The third part sets random x and y locations for each people. The last part initilize the number of days infected of each people to 0. 

Normally, to include OpenMP, all we need is to put **#pragma omp parallel** in front of each of the for loops. However, our case is a little tricky. The problem is that we are reducing the counter **our_num_infected**. Different from most parallel structure, reduction in OpenMP is pretty easy to implement. We just need to add a reduction literal,

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 300 

The problem lies on that the counter we are reducing is inside a structure, namely, the our structure. OpenMP does not support reduction to structures. Therefore, we solve this problem by first create local instance such as **our_num_infected_local** that equals to counter **our_num_infected** in our struct.

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 293

we can then, reduce to local instance,

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 306

Finally, we put local instance back to struct.

.. literalinclude:: Initialize.h  
    :language: c
    :lines: 308

We then use the same reduction method for the second part of the function. The third and Fourth part of the function does not reduce any counters, which means we don't need worry about reduction at all.

In Core.h
#########

There are four core functions inside *Core.h* file, and all of them can be parallelized using OpenMP. 

move()
******

This function is easy to parallelize because it does not perform any reduction. However, we need to specify the variables that is private to each OpenMP threads. **current_person_id** is iterator that is clearly private. **x_move_direction** and **y_move_direction** are different for every thread, which means they are private as well. 

.. literalinclude:: Core.h	
    :language: c
    :lines: 43-46 

susceptible()
*************

This function is relatively hard to parallelize because it has four counters to reduce. Luckily, we already developed our way of reducing counters in **init_array()** function, which means we can use same method in here.

Creating local instances

.. literalinclude:: Core.h	
    :language: c
    :lines: 107-112

OpenMP initialization

.. literalinclude:: Core.h	
    :language: c
    :lines: 114-119

Put local instances back to global struct

.. literalinclude:: Core.h	
    :language: c
    :lines: 173-177

infected()
**********

Similar to **susceptible()** function, we have five counters to reduce in this function.

Creating local instances

.. literalinclude:: Core.h	
    :language: c
    :lines: 202-208

OpenMP initialization

.. literalinclude:: Core.h	
    :language: c
    :lines: 210-215

Put local instances back to global struct

.. literalinclude:: Core.h	
    :language: c
    :lines: 254-259

update_days_infected()
**********************

We don't have any reduction in this function, which means that the parallelization is relatively easy. 

.. literalinclude:: Core.h	
    :language: c
    :lines: 277-279