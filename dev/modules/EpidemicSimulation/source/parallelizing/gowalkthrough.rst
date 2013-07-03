*************************************
Parallelizing with Go: Detailed Specs
*************************************

Simulating in Go
################

Structure
---------

Go is a language developed to combine features of C/C++ and Python with intuitive concurrency. If you've written either of those before, a lot of this should look familiar! If you're new to codewriting or get really stuck, check out a simple `tour`_ of the language.

This is short enough to put all the code in one file, so start with ``package main``. Then, similar to ``import`` in Python or ``#include`` in C/C++, `import`_ ``fmt`` (format - used for printing things out), ``math`` (we'll need this to find square roots), ``math/rand`` (to generate random numbers), and ``time`` (to seed our random number generator).

As you'll see if you click the link above, though, Go will give you an error if you include anything you're not using, so your code won't run until you're using all three of those libraries. If you want to test along the way, just comment out lines you're not using yet. 

And you're off! Follow the specs below to code up an epidemic in Go; if you get stuck, there are some links to check out below.

.. _tour: http://tour.golang.org/#1

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
		- ``duration``, an integer representing duration of illness, in six-hour periods

		- ``radius``, a float64 representing the radius of transmission

		- ``contagiousness``, a float64 representing the likelihood of a transmission occurring if a person is within the infection radius 

Person
``````

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
**********************

.. glossary::
	Initial parameters:
		- ``numPersons``, an integer representing the number of persons in the simulation

		- ``initialInfected``, an integer representing the number of persons who are initially infected.

		- ``numIterations``, an integer representing how many iterations the simulation runs for 

		- ``Population``, an `slice`_ (similar to an array) of type ``Person`` of size ``numPersons``, with the first ``initialInfected`` members set to Infected

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

		- Start by adding ``"sync"`` to your list of packages to import!

		- Next, seed the random object (``random.Seed()``) with the current time - otherwise it starts with the same seed every time and gives the same results. We're going to pass the Unix time value of the current time (\ ``time.Now()Unix()``\ ) - this gives us a floating-point number of seconds that changes every time the program runs.

		- Set up a loop to run the following ``numIterations`` times:

				- Loop through ``Population`` and call ``timeStep()`` on each member 

				- If the member is infected, check its position against every susceptible member of ``Population`` and find the distance between them (here is a reminder of the `distance formula`_; the square root function is ``math.Sqrt()``). 

					- If the distance between the two is less than ``radius``, use ``contagiousness`` to determine whether transmission occurs ( one possibility: transmission occurs if a random integer between zero and a hundred is less than ``contagiousness*100``)

				- Count the number of Susceptible, Infected, and Recovered members of ``Population``, and print the data to the screen (``fmt.Println()``)

.. _distance formula: http://math.about.com/library/bldistance.htm

.. _slice: http://www.golang-book.com/6#section2


Parallelizing
-------------

Background
**********

- Go provides simple options for parallelization using *goroutines*. A `goroutine`_ is a function that can run simultaneously, or concurrently, with other sections of code; they can be thought of similarly to threads, although they are not exactly synonymous. A Go program may use thousands of goroutines since they are very lightweight and are managed behind the scenes. If you write Go code thinking of goroutines as threads - considering race conditions and deadlock - you will most likely be successful.

- It is important to be able to communicate between goroutines. This is accomplished with `channels`_, which can be thought of as conveyor belts. Information is put in the channel by one goroutine and taken out by another. They are also similar to work queues in that information can be added and removed by multiple parties. Channels can convey information of any data type (including structs) and can also be buffered or unbuffered (meaning their capacity can be specified so they run asynchronously).

- The `Sync`_ library provides an easy set of tools to ensure concurrency and correct timing of goroutines. This package defines the type ``WaitGroup``, which manages the threads that should be executed loosely as a group. As goroutines are created, ``Add()`` should be called, which will increment a counter (of goroutines) within the ``WaitGroup``. When a goroutine has finished executing, call ``Done()`` to decrement the counter. ``Wait()`` will block until the counter reaches 0, signifying that all threads have finished executing. We'll use this to keep our channel open while we put infected individuals in it so that other goroutines don't start removing them too soon!

.. _channels: http://golangtutorials.blogspot.com/2011/06/channels-in-go.html

.. _Sync: http://golang.org/pkg/sync/

.. _goroutine: http://golangtutorials.blogspot.com/2011/06/goroutines.html

Give it a shot!
***************

- Start by making a WaitGroup (in library ``sync``) at global scale and a `buffered channel`_ of type ``Person`` with capacity ``numPersons`` within ``main()``\ .

- No changes should be necessary to the code for seeding the rand() function, creating the ``Population`` slice, initializing its members, creating an instance of an Infection, and infecting the first ``initialInfected`` individuals with it.

- We can't parallelize the outermost loop - each iteration depends on the one before it, so they can't run concurrently - but we can parallelize what happens within each iteration. Split the body of that loop into something like this:

.. glossary::
	Procedures:
		- ``collectInfected``

			- Arguments:
			
				- ``Population``, a pointer to your ``[]Person`` slice

				- ``i``, the integer index of the person being checked

				- ``infectedchan``, the channel for holding infected individuals

			- State change:

				- If the individual is infected, add it to the channel.

				- ``Done()`` is called on the WaitGroup to say that this goroutine has finished its task

			- Return: none

		- ``iterateThruInfected``

			- Arguments: 

					- ``Population``, a pointer to your ``Person`` slice

					- ``infectedchan``, the channel for holding infected individuals

					- the infection of your choice

			- State change:

				- `using keyword`_ ``range``, for every ``Person`` in the channel, determine whether anyone susceptible is near them and then whether transmission occurs (but make this infection process a separate function, as described below, so that a different goroutine can handle each person)

			- Return: none

		- ``infectNeighbors``

			- Arguments: 

				- ``Population``, a pointer to your ``Person`` slice

				- the infection of your choice

				- ``p``, the infected individual

			- State change: For each ``Person`` in ``Population``, if the individual is susceptible and sufficiently close to ``p``, use the disease's contagiousness to determine whether transmission occurs

			- Return: none

- After calling ``timeStep()`` on each person, call ``Add(1)`` on the WaitGroup`` and then call ``go collectInfected(i, &Population, infectedchan)`` to launch a new goroutine to handle the individual.

- After this loop, call ``Wait()`` on the WaitGroup to block until all of the goroutines examining individuals have finished.

- Next, call ``iterateThruInfected`` with the correct parameters to handle the rest of this iteration!

- Counting up the individuals in each state at the end of the simulation is a quick process, but feel free to parallelize it for as a bonus challenge.
		

.. _buffered channel: https://gobyexample.com/channel-buffering

.. _using keyword: http://golangtutorials.blogspot.com/2011/06/channels-in-go-range-and-select.html

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