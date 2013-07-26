**************
Core Functions
**************

move
****

If the person is not dead, then

.. literalinclude:: Core.h	
    :language: c
    :lines: 47

First, the function randomly picks whether the person moves left or right
or does not move in the x dimension.

The code uses (random() % 3) - 1; to achieve this. (random() % 3)
returns either 0, 1, or 2. Subtracting 1 from this produces -1, 0, or 1.
This means the person can move to the right, stay in place (0), or move
to the left (-1).

then the function randomly picks whether the person moves up or down or
does not move in the y dimension. This is similar to movement in x dimension.

.. literalinclude:: Core.h  
    :language: c
    :lines: 49-55

Next, If the person will remain in the bounds of the environment after
moving, then

We check this by making sure the person’s x location is greater than or
equal to 0 and less than the width of the environment and that the
person’s y location is greater than or equal to 0 and less than the
height of the environment. In the code, it looks like this:

.. literalinclude:: Core.h	
    :language: c
    :lines: 59-65

Finally, The function moves the person

.. figure:: img-20.png
   :align: center
   :alt: image

.. literalinclude:: Core.h  
    :language: c
    :lines: 68-71

The function is able to achieve this by simply changing values in the
**x\_locations** and **y\_locations** arrays.

susceptible
***********

For each people, the function to do the following

.. literalinclude:: Core.h	
    :language: c
    :lines: 102-103

If the person is susceptible,

.. literalinclude:: Core.h	
    :language: c
    :lines: 106

For each of the infected people or until the number of infected people nearby is 1, the function does the following

.. literalinclude:: Core.h	
    :language: c
    :lines: 112-113

If the person is within the infection radius, then

.. literalinclude:: Core.h	
    :language: c
    :lines: 116-123

Finally, the function increments the number of infected people nearby

.. literalinclude:: Core.h	
    :language: c
    :lines: 126

.. figure:: img-21.png
   :align: center
   :alt: image

This is where a large chunk of the algorithm’s computation occurs. Each
susceptible person must be computed with each infected person to
determine how many infected people are nearby each person. Two nested
loops means many computations. In this step, the computation is fairly
simple, however. The function simply increments the
**num\_infected\_nearby** variable.

Note in the code that if the number of infected nearby is greater than
or equal to 1 and we have **SHOW\_RESULTS** enabled, we increment the
**num\_infection\_attempts** variable. This helps us keep track of the
number of attempted infections, which will help us calculate the actual
contagiousness of the disease at the end of the simulation.

If there is at least one infected person nearby, and a random
number less than 100 is less than or equal to the contagiousness factor,
then

.. literalinclude:: Core.h	
    :language: c
    :lines: 139-140

Recall that the contagiousness factor is the likelihood that the disease
will be spread. We measure this as a number less than 100. For example,
if there is a 30% chance of contagiousness, we use 30 as the value of
the contagiousness factor. To figure out if the disease is spread for
any given interaction of people, we find a random number less than 100
and check if it is less than or equal to the contagiousness factor,
because this will be equivalent to calculating the odds of actually
spreading the disease (e.g. there is a 30% chance of spreading the
disease and also a 30% chance that a random number less than 100 will be
less than or equal to 30).

The function changes the state to infected

.. literalinclude:: Core.h	
    :language: c
    :lines: 143

The function updates the counters

.. literalinclude:: Core.h	
    :language: c
    :lines: 144-146

.. figure:: img-22.png
   :align: center
   :alt: image

These steps are as simple as updating the **states** array by
**states[my\_current\_person\_id] = INFECTED**, incrementing the
**num\_infected** variable, and decrementing the **num\_susceptible**
variable.

Note in the code that if the infection succeeds and we have
**SHOW\_RESULTS** enabled, we increment the **num\_infections variable**.
This helps us keep track of the actual number of infections, which will
help us calculate the actual contagiousness of the disease at the end of
the simulation.

infected
********

For each people, the function to do the following

.. literalinclude:: Core.h	
    :language: c
    :lines: 178-179

If the person is infected and has been for the full duration of the disease, then

.. literalinclude:: Core.h	
    :language: c
    :lines: 183-185

Note in the code that if we have **SHOW\_RESULTS** enabled, we increment the
**num\_recovery\_attempts** variable. This helps us keep track of the
number of attempted recoveries, which will help us calculate the actual
deadliness of the disease at the end of the simulation.

.. literalinclude:: Core.h	
    :language: c
    :lines: 189

If a random number less than 100 is less than the deadliness factor, then

.. literalinclude:: Core.h	
    :language: c
    :lines: 193

The function changes the person’s state to dead

.. literalinclude:: Core.h	
    :language: c
    :lines: 196

The function updates the counters

.. literalinclude:: Core.h	
    :language: c
    :lines: 197-199

.. figure:: img-23.png
   :align: center
   :alt: image

This step is effectively the same as function susceptible, considering deadliness instead of contagiousness. The difference here is the following step:

Otherwise,

The function changes the person’s state to immune

.. literalinclude:: Core.h	
    :language: c
    :lines: 209

The function updates the counters

.. literalinclude:: Core.h	
    :language: c
    :lines: 210-212

.. figure:: img-24.png
   :align: center
   :alt: image

If deadliness fails, then immunity succeeds.

Note in the code that if the person dies and we have **SHOW\_RESULTS**
enabled, we increment the **num\_deaths** variable. This helps us keep
track of the actual number of deaths, which will help us calculate the
actual deadliness of the disease at the end of the simulation.

.. literalinclude:: Core.h	
    :language: c
    :lines: 200-203

update_days_infected
********************

For each people, the function to do the following

.. literalinclude:: Core.h	
    :language: c
    :lines: 233-234

If the person is infected, then

.. literalinclude:: Core.h	
    :language: c
    :lines: 237

Increment the number of days the person has been infected

.. literalinclude:: Core.h	
    :language: c
    :lines: 240

.. figure:: img-25.png
   :align: center
   :alt: image

This is achieved by incrementing each member of the
**num\_days\_infected** array, which can be done as follows:
**num\_days\_infected[my\_current\_person\_id]++**