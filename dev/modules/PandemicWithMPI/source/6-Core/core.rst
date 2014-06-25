**************
Core Functions
**************

move()
******

For of the each process's people, this function moves them around randomly.

For everyone handled by this process,

.. literalinclude:: Core.h  
    :language: c
    :lines: 46-47

If the person is not dead, then

.. literalinclude:: Core.h	
    :language: c
    :lines: 50

First, The thread randomly picks whether the person moves left or right or does not move in the x dimension.

The code uses (random() % 3) - 1; to achieve this. (random() % 3) returns either 0, 1, or 2. Subtracting 1 from this produces -1, 0, or 1. This means the person can move to the right, stay in place (0), or move to the left (-1).

.. literalinclude:: Core.h  
    :language: c
    :lines: 52-54

The thread then randomly picks whether the person moves up or down or does not move in the y dimension. This is similar to movement in x dimension.

.. literalinclude:: Core.h  
    :language: c
    :lines: 56-58

Next, we need to make sure that the person will remain in the bounds of the environment after moving. We check this by making sure the person’s x location is greater than or equal to 0 and less than the width of the environment and that the person’s y location is greater than or equal to 0 and less than the height of the environment. In the code, it looks like this:

.. literalinclude:: Core.h  
    :language: c
    :lines: 62-69

Finally, The thread moves the person

.. figure:: img-20.png
   :align: center
   :alt: image

The thread is able to achieve this by simply changing values in the
**our\_x\_locations** and **our\_y\_locations** arrays.

.. literalinclude:: Core.h  
    :language: c
    :lines: 71-73

susceptible()
*************

For of the each process's people, this function handles those that are ssusceptible by deciding whether or not they should be marked infected.

For everyone handled by this process,

.. literalinclude:: Core.h	
    :language: c
    :lines: 121-122

If the person is susceptible,

.. literalinclude:: Core.h	
    :language: c
    :lines: 125

For each of the infected people (received earlier from all processes) or until the number of infected people nearby is 1, the thread does the following

.. literalinclude:: Core.h	
    :language: c
    :lines: 131-132

If this person is within the infection radius, 

.. literalinclude:: Core.h	
    :language: c
    :lines: 135-142

then, the function increments the number of infected people nearby

.. literalinclude:: Core.h	
    :language: c
    :lines: 145

.. figure:: img-21.png
   :align: center
   :alt: image

This is where a large chunk of the algorithm’s computation occurs. Each susceptible person must be computed with each infected person to determine how many infected people are nearby each person. Two nested loops means many computations. In this step, the computation is fairly simple, however. The thread simply increments the **my\_num\_infected\_nearby** variable.

Note in the code that if the number of infected nearby is greater than or equal to 1 and we have **SHOW\_RESULTS** enabled, we increment the **our\_num\_infection\_attempts** variable. This helps us keep track of the number of attempted infections, which will help us calculate the actual contagiousness of the disease at the end of the simulation.

.. literalinclude:: Core.h  
    :language: c
    :lines: 149-153

If there is at least one infected person nearby, and a random number less than 100 is less than or equal to the contagiousness factor, then

.. literalinclude:: Core.h	
    :language: c
    :lines: 158-159

Recall that the contagiousness factor is the likelihood that the disease will be spread. We measure this as a number less than 100. For example, if there is a 30% chance of contagiousness, we use 30 as the value of the contagiousness factor. To figure out if the disease is spread for any given interaction of people, we find a random number less than 100 and check if it is less than or equal to the contagiousness factor, because this will be equivalent to calculating the odds of actually spreading the disease (e.g. there is a 30% chance of spreading the disease and also a 30% chance that a random number less than 100 will be less than or equal to 30).

The thread changes this person state to infected

.. literalinclude:: Core.h	
    :language: c
    :lines: 161-162

The thread updates the counters

.. literalinclude:: Core.h	
    :language: c
    :lines: 163-165

.. figure:: img-22.png
   :align: center
   :alt: image

Note in the code that if the infection succeeds and we have **SHOW\_RESULTS** enabled, we increment the **our\_num\_infections variable**. This helps us keep track of the actual number of infections, which will help us calculate the actual contagiousness of the disease at the end of the simulation.

.. literalinclude:: Core.h  
    :language: c
    :lines: 166-169

infected()
**********

For of the each process's people, this function to handles those that are infected by deciding whether they should be marked immune or dead.

For everyone handled by this process,

.. literalinclude:: Core.h	
    :language: c
    :lines: 217-218

If the person is infected and has been for the full duration of the disease, then

.. literalinclude:: Core.h	
    :language: c
    :lines: 222-223

Note in the code that if we have **SHOW\_RESULTS** enabled, we increment the **our\_num\_recovery\_attempts** variable. This helps us keep track of the number of attempted recoveries, which will help us calculate the actual deadliness of the disease at the end of the simulation.

.. literalinclude:: Core.h	
    :language: c
    :lines: 225-228

If a random number less than 100 is less than the deadliness factor, 

.. literalinclude:: Core.h	
    :language: c
    :lines: 231

then, the thread changes the person’s state to dead

.. literalinclude:: Core.h	
    :language: c
    :lines: 234

and then the thread updates the counters

.. literalinclude:: Core.h	
    :language: c
    :lines: 235-237

.. figure:: img-23.png
   :align: center
   :alt: image

This step is effectively the same as function susceptible, considering deadliness instead of contagiousness. The difference here is the following step:

if a random number less than 100 is less than the deadliness factor, the thread changes the person’s state to immune

.. literalinclude:: Core.h	
    :language: c
    :lines: 246-247

The thread updates the counters

.. literalinclude:: Core.h	
    :language: c
    :lines: 248-250

.. figure:: img-24.png
   :align: center
   :alt: image

If deadliness fails, then immunity succeeds.

Note in the code that if the person dies and we have **SHOW\_RESULTS** enabled, we increment the **our\_num\_deaths** variable. This helps us keep track of the actual number of deaths, which will help us calculate the actual deadliness of the disease at the end of the simulation.

.. literalinclude:: Core.h	
    :language: c
    :lines: 238-241

update_days_infected()
**********************

For of the each process's people, this function to handles those that are infected by increasing the number of days infected.

For everyone handled by this process,

.. literalinclude:: Core.h	
    :language: c
    :lines: 281-282

If the person is infected, 

.. literalinclude:: Core.h	
    :language: c
    :lines: 285

then, the function increment the number of days the person has been infected

.. literalinclude:: Core.h	
    :language: c
    :lines: 288

.. figure:: img-25.png
   :align: center
   :alt: image