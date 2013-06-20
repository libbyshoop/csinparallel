*************************************
Parallelizing with Go: Detailed Specs
*************************************

Simulating in Go
################

Structure
---------

Go is a language developed to combine features of C/C++ and Python with intuitive concurrency. If you've written either of those before, a lot of this should look familiar!

This is short enough to put all the code in one file, so start with ``package main``. Then, similar to ``import`` in Python or ``#include`` in C/C++, `import`_ ``fmt`` (format - used for printing things out), ``math`` (we'll need this to find square roots), and ``math/rand`` (to generate random numbers).

As you'll see if you click the link above, though, Go will give you an error if you include anything you're not using, so your code won't run until you're using all three of those libraries. If you want to test along the way, just comment out lines you're not using yet. 

And you're off! Follow the specs below to code up an epidemic in Go; if you get stuck, there are some links to check out below.

.. _import: http://golangtutorials.blogspot.com/2011/05/early-syntax-errors-and-other-minor.html

At Global Scope
***************

.. glossary::
	width & height:
		**constant** integers representing the dimensions of your simulated world - these are parameters to play with!

.. glossary::
	list of states:
		Like enums in C++, or like using range() in Python to refer to an item from a list by its position in the list, Go allows you to hook words up with numbers with the keyword ``iota`` (see `example`_).

.. _example: http://rosettacode.org/wiki/Enumerations#Go

Structs
*******

Go doesn't have classes with methods the way Python and C++ do, but it does have structs which can have functions associated with them. Structs are declared ``type <name> struct {}``, with your chosen name in place of ``<name>``; variables are likewise declared by name and *then* type (i.e. ``duration int``).

Infection
`````````

Represents a particular disease.

.. glossary::
	Variables:
		- ``duration``, an integer representing duration of illness

		- ``radius``, a float64 representing the radius of transmission

		- ``contagiousness``, a float64 representing the likelihood of a transmission occurring if a person is within the infection radius 

Struct Person
-------------

Represents an individual human being.

.. glossary::
	State Variables:
		- ``x`` and ``y``, two integers representing the Person’s 2D position

		- ``state``, a integer identifying what state of health the Person is in

		- ``infectedPeriod``, an integer representing the units of time required before a Person makes a full recovery. 

Methods associated with a struct are written like this: 

	.. code-block:: go
		:linenos:

		func (p *Person) updateState(s int) bool {
			p.state = s
		}

	.. note::
		After ``func`` the struct to which the function is associated in included in parentheses. The asterisk before ``Person`` indicates that the object itself should be acted on rather than a copy of it - this looks like a C++ pointer but is more similar to passing by reference. If there is a return value, its type is indicated at the end of the function declaration; if the function is void, no return type is given. 

.. glossary::
	Associated methods:
		- ``init``
			- Since Go doesn't have constructors, the easiest way to initialize instances of a struct seems to be with a function. 

			- Arguments: none

			- State change: ``x`` is set to a random integer between zero and ``width`` and ``y`` is set to a random integer between zero and height ``height``; ``state`` is set to Susceptible and ``infectedPeriod`` to 0

		- ``isInfected`` 
			- Arguments: none

			- State change: none

			- Return: a boolean value which is ``true`` if the person is Infected and ``false`` otherwise

		- ``isSusceptible`` 
			- Arguments: none

			- State change: none

			- Return: a boolean value which is ``true`` if the person is Susceptible and ``false`` otherwise

		- ``updateState`` 
			- Arguments: ``s int``

			- State change: ``state`` is set to ``s``

			- Return: none

		- ``infectWith`` 
			- Arguments: ``i Infection``

			- State change: ``state`` is set to Infected and ``infectedPeriod`` is set to ``i``\ 's ``duration``

			- Return: none

		- ``move`` 
			- Arguments: none

			- State change: location is randomly changed by 0, 1, or 2 units in the x direction and y direction

			.. warning:: The mod function has no effect on negative numbers. This is a problem if the position variables become negative. An easy way to solve this problem is to add width before you mod by width. Your code might look something like ``x = (x + (rand.Intn(5) - 2 + width) % width;`` (and similarly with ``y``).

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

		- ``Population``, an array of ``Person`` of size ``numPersons``, with the first ``initialInfected`` members set to Infected

		.. note:: 
			You'll probably want to use a loop to call ``init()`` on each of the members and then to infect the first ``initialInfected`` of them. Go doesn't have ``while`` loops, but the syntax of ``for`` loops is like this (lack of parentheses; ``:=`` for initializing a new variable, and ``i++`` rather than ``++i``):
	
				.. code-block:: go
					:linenos:

					for i := 0; i<initialInfected; i++ { 
						...
					}

		- ``disease``, an instance of an infection with the parameters of your choice

.. glossary::
	Procedures:
		- Set up a loop to run the following ``numIterations`` times:

				- Loop through ``Population`` and call ``timeStep()`` on each member 

				- If the member is infected, check its position against every susceptible member of ``Population`` and find the distance between them (here is a reminder of the `distance formula`_; the square root function is ``math.Sqrt()``). 

					- If the distance between the two is less than ``radius``, use ``contagiousness`` to determine whether transmission occurs ( one possibility: transmission occurs if a random integer between zero and a hundred is less than ``contagiousness*100``)

				- Count the number of Susceptible, Infected, and Recovered members of ``Population``, and print the data to the screen (``fmt.Println()``)

.. _distance formula: http://math.about.com/library/bldistance.htm


Parallelizing
-------------

- Go provides simple options for parallelization using goroutines. Briefly, a goroutine is a function that can run simultaneously, or concurrently, with other sections of code. Goroutines can be thought of similarly to threads, although they are not exactly synonymous. A Go program may use thousands of goroutines since they are very lightweight and are managed behind the scenes. If a programmer writes code thinking of goroutines as threads considering race conditions and deadlock, they will most likely be successful

- When using goroutines, it is important to be able to communicate between goroutines. This is accomplished by channels which can be thought of as conveyor belts. Information is put in the channel by one goroutine and taken out by another. They can also be compared to work queues since information can be added to and removed from by multiple parties. Channels can convey any data type of information including structs and are also used buffered or unbuffered.

- The Sync library provides an easy method to ensure concurrency and timing of goroutines. This package defines the type WaitGroup which manages the threads that should be executed loosely as a group. This is accomplished by a few methods defined for a WaitGroup such as add() and done(). As goroutines are created, add() should be called which will increment a counter (of goroutines) within the WaitGroup. When a goroutine has finished executing, call Done() which will decrement the counter. There is a blocking method called Wait() that will block until the counter reaches 0 signifying that all threads have finished executing. Other options include Locks which mimic a physical lock that can be open and closed by goroutines, but we did not pursue these for our simulation

- To parallelize the go simulation we used several (in fact, hundreds) goroutines to divide up the computation of the simulation. The parallelization can be likened to parallelizing loops with OpenMP. Within a for loop, create a new goroutine to execute the body of the loop for each iteration. The body of the loop will need to be defined in a function and the appropriate variables pass (by reference if needed).

- To communicate and share information between goroutines, create a channel used to share and store infected people. As goroutines identify people as infected, they add the person to the channel. Different goroutines should later remove people from the channel and compare the distance between all susceptible people and this infected individual. 

- Unfortunately if a channel is empty for too long, it will close. To ensure all infected people are added to the channel a blocking mechanism is necesary. Use a WaitGroup and add all goroutines that are involved in searching for infected persons. Call Wait() in the WaitGroup before beginning to remove people from the channel. When the last of these goroutines have finished their search and have called Done(), the call to Wait() will finish blocking and the program will continue execution.

Resources
---------

- Official Go `source code`_

- If you've got some C++ knowledge, `these hints`_ on Go for C++ programmers will probably help.

- For Python programmers, `here`_ some slides from a relevant talk (link opens a PDF).

- To just start from scratch (or to look up a particular topic), check out this `introductory book`_.

.. _these hints: https://code.google.com/p/go-wiki/wiki/GoForCPPProgrammers

.. _source code: http://golang.org/pkg/

.. _here: http://s3.amazonaws.com/golangweekly/go_for_pythonistas.pdf

.. _introductory book: http://www.golang-book.com/