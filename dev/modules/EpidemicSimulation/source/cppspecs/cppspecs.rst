**************************************
A Sequential Implementation: C++ Specs
**************************************

Based on the explanation in the previous section and the specifications given here, try implementing a sequential version of this model yourself in C++. In the next sections, we'll look at using parallelism to speed it up!

At Global Scope
###############

.. glossary::
	width & height:
		constant integers representing the dimensions of your simulated world.

.. glossary::
	State:
		An `enum`_ for the states of health a Person can have with respect to the disease, with three options:
		
		- Susceptible
	
		- Infected

		- Recovered


Classes
#######

Class Infection
---------------

Represents a particular disease.

.. glossary::
	State Variables:
		- ``duration``, an integer representing duration of illness, in six-hour periods

		- ``radius``, a float representing the radius of transmission

		- ``contagiousness``, a float representing the likelihood of a transmission occurring if a person is within the infection radius 

.. glossary::
	 Constructor (optional):
	 	- ``Infection``
	 		- Arguments: int ``d``, float ``r``, float ``c``
		
			- State change: ``duration``, ``radius``, and ``contagiousness`` set to ``d``, ``r``, and ``c``, respectively

Class Person
------------

Represents an individual human being.

.. glossary::
	State Variables:
		- ``x`` and ``y``, two integers representing the Person’s 2D position

		- ``state``, a State variable identifying what state of health the Person is in

		- ``infectedPeriod``, an integer representing the units of time required before a Person makes a full recovery. 

.. glossary::
	Constructor:
		- ``Person``
			- Arguments: none

			- State change: ``x`` is set to a random integer modulo ``width`` and ``y`` is set to a random integer mod ``height``; ``state`` is set to Susceptible and ``infectedPeriod`` to 0


.. glossary::
	Methods:
		- ``isInfected`` 
			- Arguments: none

			- State change: none

			- Return: ``true`` if the person is Infected; ``false`` otherwise

		- ``isSusceptible`` 
			- Arguments: none

			- State change: none

			- Return: ``true`` if the person is Susceptible; ``false`` otherwise

		- ``updateState`` 
			- Arguments: ``State s``

			- State change: ``state`` is set to ``s``

			- Return: none

		- ``infectWith`` 
			- Arguments: ``Infection i``

			- State change: ``state`` is set to Infected and ``infectedPeriod`` is set to ``i``\ 's ``duration``

			- Return: none

		- ``move`` 
			- Arguments: none

			- State change: location is randomly changed by 0, 1, or 2 units in the x direction and y direction

			.. warning:: The modulo function in C++ has no effect on negative numbers. This is a problem if the position variables become negative. An easy way to solve this problem is to add width before you mod by width. Your code might look something like ``x = (x + (rand() % 5) - 2 + width) % width;`` (and similarly with ``y``).

			- Return: none

		- ``timeStep``
			- Arguments: none

			- State change: ``move()`` is called. If ``infectedPeriod`` is greater than zero, it is decremented; if it is zero and ``state`` is Infected, ``state`` is set to Recovered.

			- Return: none

Simulating an Epidemic
######################

.. glossary::
	Initial parameters:
		- ``numPersons``, an integer representing the number of persons in the simulation

		- ``initialInfected``, an integer representing the number of persons who are initially infected.

		- ``numIterations``, an integer representing how many iterations the simulation runs for

		- ``Population``, a dynamically-allocated array of ``Person`` of size ``numPersons``, with the first ``initialInfected`` members set to Infected

		- ``disease``, an instance of an infection with the parameters of your choice

.. glossary::
	Procedures:
		- Set up your preferred type of loop to run the following ``numIterations`` times:

			- For each member of ``Population``	
				
				- Call ``timeStep()``

				- If the member is infected, check its position against every susceptible member of ``Population``, and, if the distance between the two is less than ``radius``, use ``contagiousness`` to determine whether transmission occurs.

		- Count the number of Susceptible, Infected, and Recovered members of ``Population``, and print them out to compare to the initial parameters.


.. _enum: http://www.cplusplus.com/doc/tutorial/other_data_types/

