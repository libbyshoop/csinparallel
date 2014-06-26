*******************
Infection Functions
*******************

find_infected
*************

This function determines the **x_location** and **y_location** of all the infected people.

.. figure:: img-16.png
   :align: center
   :alt: image

We have already set the states of the infected people and the positions
of all the people, but we need to specifically set the positions of the
infected people and store them in the **infected\_x\_locations** and
**infected\_y\_locations** arrays. We do this by marching through the
**states** array and checking whether the state at each cell is
**INFECTED**. If it is, we add the locations of the current infected person
from the **x\_locations** and **y\_locations** arrays to the
**infected\_x\_locations** and **infected\_y\_locations** arrays. We
determine the ID of the current infected person using the
**current\_infected\_person** variable:

.. literalinclude:: Infection.h	
    :language: c
    :lines: 23-35