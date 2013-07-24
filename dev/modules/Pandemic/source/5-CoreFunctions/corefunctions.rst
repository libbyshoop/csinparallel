**************
Core Functions
**************

move
****

If the person is not dead, then

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 478

First, The thread randomly picks whether the person moves left or right
or does not move in the x dimension.

The code uses (random() % 3) - 1; to achieve this. (random() % 3)
returns either 0, 1, or 2. Subtracting 1 from this produces -1, 0, or 1.
This means the person can move to the right, stay in place (0), or move
to the left (-1).

The thread randomly picks whether the person moves up or down or
does not move in the y dimension. This is similar to movement in x dimension.

Next, If the person will remain in the bounds of the environment after
moving, then

We check this by making sure the person’s x location is greater than or
equal to 0 and less than the width of the environment and that the
person’s y location is greater than or equal to 0 and less than the
height of the environment. In the code, it looks like this:

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 490-497

Finally, The thread moves the person

.. figure:: img-20.png
   :align: center
   :alt: image

The thread is able to achieve this by simply changing values in the
**our\_x\_locations** and **our\_y\_locations** arrays.

susceptible
***********

For each of the process’s people, each process spawns threads to do the following

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 547-548

If the person is susceptible,

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 551

For each of the infected people (received earlier from all processes) or until the number of infected people nearby is 1, the thread does the following

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 557-558

If person 1 is within the infection radius, then

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 561-568

The thread increments the number of infected people nearby

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 571

.. figure:: img-21.png
   :align: center
   :alt: image

This is where a large chunk of the algorithm’s computation occurs. Each
susceptible person must be computed with each infected person to
determine how many infected people are nearby each person. Two nested
loops means many computations. In this step, the computation is fairly
simple, however. The thread simply increments the
**my\_num\_infected\_nearby** variable.

Note in the code that if the number of infected nearby is greater than
or equal to 1 and we have **SHOW\_RESULTS** enabled, we increment the
**our\_num\_infection\_attempts** variable. This helps us keep track of the
number of attempted infections, which will help us calculate the actual
contagiousness of the disease at the end of the simulation.

If there is at least one infected person nearby, and a random
number less than 100 is less than or equal to the contagiousness factor,
then

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 583-584

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

The thread changes person1’s state to infected

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 587

The thread updates the counters

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 590-595

.. figure:: img-22.png
   :align: center
   :alt: image

These steps are as simple as updating the **our\_states** array by
**our\_states[my\_current\_person\_id] = INFECTED**, incrementing the
**our\_num\_infected** variable, and decrementing the **our\_num\_susceptible**
variable.

Note in the code that if the infection succeeds and we have
**SHOW\_RESULTS** enabled, we increment the **our\_num\_infections variable**.
This helps us keep track of the actual number of infections, which will
help us calculate the actual contagiousness of the disease at the end of
the simulation.

infected
********

For each of the process’s people, each process spawns threads to do the following

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 641-642

If the person is infected and has been for the full duration of the disease, then

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 646-647

Note in the code that if we have **SHOW\_RESULTS** enabled, we increment the
**our\_num\_recovery\_attempts** variable. This helps us keep track of the
number of attempted recoveries, which will help us calculate the actual
deadliness of the disease at the end of the simulation.

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 649-651

If a random number less than 100 is less than the deadliness factor, then

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 654

The thread changes the person’s state to dead

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 657

The thread updates the counters

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 659-660

.. figure:: img-23.png
   :align: center
   :alt: image

This step is effectively the same as function susceptible, considering deadliness
instead of contagiousness. The difference here is the following step:

Otherwise,

The thread changes the person’s state to immune

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 670

The thread updates the counters

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 670-673

.. figure:: img-24.png
   :align: center
   :alt: image

If deadliness fails, then immunity succeeds.

Note in the code that if the person dies and we have **SHOW\_RESULTS**
enabled, we increment the **our\_num\_deaths** variable. This helps us keep
track of the actual number of deaths, which will help us calculate the
actual deadliness of the disease at the end of the simulation.

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 662-664

updateDays
**********

For each of the process’s people, each process spawns threads to do the following

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 702-703

If the person is infected, then

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 706

Increment the number of days the person has been infected

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 709

.. figure:: img-25.png
   :align: center
   :alt: image

This is achieved by incrementing each member of the
**our\_num\_days\_infected** array, which can be done as follows:
**our\_num\_days\_infected[my\_current\_person\_id]++**