******************************************
Parallelizing Sequential Go Code: Beginner
******************************************

Simulating in Go
################

Let's Go!
---------

Go is a language developed to combine features of C/C++ and Python with intuitive concurrency.

Go code for simulating this epidemic is available at :download:`epidemic.go <code/epidemic.go>`. Read through the code and the comments, then follow the directions below to take advantage of the simplicity of parallelizing in Go.


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