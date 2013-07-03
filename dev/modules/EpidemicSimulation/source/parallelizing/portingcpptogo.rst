**********************
Porting C++ Code to Go
**********************

Simulating in Go
################

Go is a C-like language developed to do concurrency intuitively. Below are some key differences between the two as well as an introduction to parallelizing Go code (and resources to figure out the rest!).

Code Structure
--------------
There are not classes in the Go language, but there are structs which can be instantiated like objects in C++. Special functions called *methods* which are defined outside of the struct can be called on the instance of the struct.

Such a function would look like this:

	.. code-block:: go
		:linenos:

		func (p *Person) updateState(s int) {
			p.state = s
		}


After ``func`` the struct to which the function is associated in included in parentheses. **If you wish to modify the instance of the struct, use a pointer.** 

Go doesn't have ``while`` loops, but the syntax of ``for`` loops is like this (lack of parentheses; ``:=`` for initializing a new variable, and ``i++`` rather than ``++i``):
	
	.. code-block:: go
		:linenos:

		for i := 0; i<initialInfected; i++ { 
			...
		}

Finally, there are not enums, so declare const global variables instead.

For further help, check out `these hints`_ on Go for C++ programmers. You also may want to consult the `documentation`_, this `brief tutorial`_, a `set of examples by topic`_, and a longer `introductory book`_. These resources and a search engine will have you writing Go code in no time.

.. _these hints: https://code.google.com/p/go-wiki/wiki/GoForCPPProgrammers

.. _documentation: http://golang.org/pkg/

.. _brief tutorial: http://golangtutorials.blogspot.com/2011/05/table-of-contents.html

.. _set of examples by topic: https://gobyexample.com/

.. _introductory book: http://www.golang-book.com/

Parallelizing
-------------

- Go provides simple options for parallelization using *goroutines*. A `goroutine`_ is a function that can run simultaneously, or concurrently, with other sections of code; they can be thought of similarly to threads, although they are not exactly synonymous. A Go program may use thousands of goroutines since they are very lightweight and are managed behind the scenes. If you write Go code thinking of goroutines as threads - considering race conditions and deadlock - you will most likely be successful.

- It is important to be able to communicate between goroutines. This is accomplished with `channels`_, which can be thought of as conveyor belts. Information is put in the channel by one goroutine and taken out by another. They are also similar to work queues in that information can be added and removed by multiple parties. Channels can convey information of any data type (including structs) and can also be buffered or unbuffered (meaning their capacity can be specified so they run asynchronously).

- The `Sync`_ library provides an easy set of tools to ensure concurrency and correct timing of goroutines. This package defines the type ``WaitGroup``, which manages the threads that should be executed loosely as a group. As goroutines are created, ``Add()`` should be called, which will increment a counter (of goroutines) within the ``WaitGroup``. When a goroutine has finished executing, call ``Done()`` to decrement the counter. ``Wait()`` will block until the counter reaches 0, signifying that all threads have finished executing.

	.. note :: 
		Go's Sync library also includes Locks that goroutines can open and close. Feel free to pursue them as an extension; our focus here is on thread blocking and waiting, which are more in line with MPI/OpenMP concurrency styles.

- To parallelize the Go simulation we can use goroutines to divide up the computation, similar to parallelizing loops with OpenMP. Within a ``for`` loop, create a new goroutine to execute the body of the loop for each iteration. The body of the loop will need to be defined in a function and the appropriate variables passed (by reference if needed). Note that each iteration depends on the one before, so entire iterations cannot be run in parallel, but a separate goroutine can handle calling ``timeStep()`` on each ``Person`` and determining whether he is infected, and other goroutines can handle infecting the neighbors of each infected ``Person``.

- To communicate between your goroutines, create a channel used to share and store infected ``Person``\ s. As goroutines identify individuals as infected, they add them to the channel. In a separate function, different goroutines should later (`using keyword`_ ``range``) each remove an individual from the channel and compare the distance between all susceptible others and the infected individual. 

	.. warning::
		If a channel is empty for too long, it will close. To prevent people from being removed faster than they're being put in (and therefore the channel becoming empty and closing before all infected people are added), a blocking mechanism is necesary. Before calling the (parallelized) function that puts infected people into the channel, ``Add(1)`` to your WaitGroup, and then before beginning to remove individuals from the channel, call ``Wait()`` to ensure that the removal step doesn't begin before the inserting step has ended. Each goroutine checking to see if a ``Person`` is infected should call ``Done()`` before returning. Once they all have, the program will resume execution with the instruction after the ``Wait()``.


.. _goroutine: http://golangtutorials.blogspot.com/2011/06/goroutines.html
.. _channels: http://www.golang-book.com/10#section2
.. _Sync: http://golang.org/pkg/sync/
.. _using keyword: http://golangtutorials.blogspot.com/2011/06/channels-in-go-range-and-select.html