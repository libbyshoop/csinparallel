.. Pi Using Numerical Integration: Chapel documentation master file, created by
   sphinx-quickstart on Mon Sep  8 19:41:40 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pi Using Numerical Integration: Chapel
======================================

.. toctree::
   :maxdepth: 1

Chapel is an open source parallel programming language designed and developed by Cray Inc in colloboration with academia, computing centers, and industry. Chapel supports parallelism in many different ways, so we will show a sequential solution and compare it to various parallel solutions, some of which will work better than others.

You can download the entire :download:`program <pi_integration.chpl>` so that you can test the differences between different solutions. You can download and install the Chapel compiler from their `website <http://chapel.cray.com/index.html>`_. Once you have set up the Chapel compiler, you can compile the code by typing the following into the command line:

 .. code-block:: bash
	
	chpl -o pi pi_integration.chpl

Sequential Solution
*******************

First we need to set a few global variables that most of the proceedures (equivalent of functions/methods) will use. When we run the code in command line, we can add flags to change the value of the ``config`` variables. The last section of the program has a series of if statements which run a different proceedure depending on which algorithm name you put in for the ``method`` variable.


 .. code-block:: java

   config var numRect: int = 10000000;    // Specifies the number of rectangles to calculate
   config var numThreads: int = 8;        // Specifies the number of threads to run
   config var feedback: bool = false;     // Toggles text displaying partial sums or global sum for certain proceedures. 
   config var method: string = "reduceThread";  // Chooses the algorithm to calculate pi. Options include:
       	   	   	    	          // linear, forall, forallSync, beginRace, begin, reduceRect, reduceThread 	       	    	       	  
       	   	   	    	         	       
   var globalSum: real = 0.0;
   var lock: sync bool;

Then we make a proceedure that uses globalSum to keep track of the area of all the rectangles in the unit circle.

 .. code-block:: java

    proc areaLinear() {  

      // Calculates the area under half a curve without any parallel processing.

        var width: real = 2.0 / numRect;
        var x: real = -1 + width/2;
        for i in 1..numRect-1 {
            x = -1 + ( i + 0.5) * width;
	    globalSum += sqrt(1.0 - x*x) * width;	    
        }
    }
    
    if (method == "linear") {
        areaLinear();     
    }

The program ends by doubling ``globalSum`` and printing the result along with the result of a timer that keeps track of the number of seconds it takes during the time that any of the algorithms run.
    
 .. code-block:: java
    
    var pi: real = globalSum*2.0;
    writeln("This program estimates pi as ", pi);
    writeln("The operation took ", timer.elapsed(), " seconds.");
    
To see the sequential proceedure, run the executable using the following:

 .. code-block:: bash
		
	./pi --method=linear

It should accurately estimate the first six digits of pi.

``forall`` Loop
********************

Now we will look at an example of an algorithm that attempts to calculate pi with one of Chapel's features for parallel processing. 
    
 .. code-block:: java

    proc areaForall() { 

      /* Calculates the area under half a curve using a forall loop. 
	 Notice that it has a race condition, and has different levels
	 of accuracy and precision depending on the size of numRect.
      */

	 var width: real = 2.0 / numRect;
	 var x: real = -1 + width/2;
	 forall i in 1..numRect-1 {
		x = -1 + ( i + 0.5) * width;
		globalSum += sqrt(1.0 - x*x) * width;	    
	 }
    }
    
    if (method == "forall") {
	areaForall();
    }
    
Chapel's ``forall`` loop will use a new thread for each core of the processor to run the iterations of the loop in parallel. However, this creates a race condition as multiple threads may try to update ``globalSum`` at the same time, so it will not accurately estimate pi. The algorithm gives varying levels of accuracy depending on how many rectangles it tries to calculate for. You can experiment with different values for ``numRect`` to see what values give the closest accuracy, and think about why some values would get closer to pi than others. To change ``numRect``, run the executable using the following flags and change the number that ``numRect`` is assigned to.

 .. code-block:: bash

	./pi --method=forall --numRect=100000

We can stop this race condition from happening by using thread safe variable instead of ``globalSum``. We can use a ``sync`` variable, which stores a normal value and a second boolean value that we can set to ``empty`` or ``full``. Threads cannot access a full ``sync`` variable. Assigning a ``sync`` variable a value sets it to ``full``, while assigning its value to another ``sync`` variable sets it to ``empty``. So in this next example, threads can safely change the value of ``sum``, and then we can set ``globalSum`` to the value of ``sum`` after the ``forall`` loop.

 .. code-block:: java

    proc areaForallSync() { 

      /* Calculates the area under half a curve using a forall loop. 
	 Notice that it has a race condition, and has different levels
	 of accuracy and precision depending on the size of numRect.
      */

	 var sum: sync real = 0.0;
	 var width: real = 2.0 / numRect;
	 var x: real = -1 + width/2;
	 forall i in 1..numRect-1 {
		x = -1 + ( i + 0.5) * width;
		sum += sqrt(1.0 - x*x) * width;	    
	 }
	 globalSum = sum;
    }

    if (method == "forallSync") {
	areaForallSync();
    }

When you run this proceedure, it should correctly estimate pi, but you will notice that it runs `much` more slowly than the sequential solution (you may want to try running it with a lower value of numRect). When each thread has to wait for the other threads to finish accessing ``sum``, the program doesn't really process any information in parallel and we lose even more efficiency because of overhead costs. So how can we add up each thread's calculations without them interfering with each other? We will show different ways to do this with the following algorithms.

``begin`` Statements
********************

The next algorithm uses ``begin`` statements to start new threads which add to their local variable ``partialSum`` and later add their partial summations to ``globalSum``. Different threads adding to ``globalSum`` does create a potential to cause a race condition, but each thread will only modify it once so they have an extremely low chance of modifiying it at the same time. We can easily make this algorithm thread-safe, so we should do so just in case. If this program only used this algorithm, we could just make ``globalSum`` a ``sync`` variable, but since other algorithms use ``globalSum`` we don't want to change it. Instead, we can declare a new ``sync`` boolean variable outside of the algorithm, so that we can set it to ``full`` before we add to ``globalSum`` and then empty it afterwards.   

 .. code-block:: java

    var lock: sync bool;

    proc areaBegin(init) { 

	// Calculates the area under the curve for 1/numThreads worth 
	// of rectangles and adds the result to globalSum.

	var width: real = 2.0 / numRect;
	var partialSum: real = 0.0;
	var x: real = -1 + width/2;
	var i: int = init;
	do {
	x = -1 + ( i + 0.5) * width;
	    partialSum += sqrt(1.0 - x*x) * width;	    
	    i += numThreads;
	}   while (i < numRect-1);
	lock = true; // set sync variable to full
	globalSum += partialSum;
	var unlock = lock; // empty the sync variable to allow the program to move on
	if (feedback) {writeln("Thread: ", init, "   globalSum: ", globalSum);} 
    }

    if (method == "beginRace") {
	for i in 1..numThreads {
	    begin areaBegin(i);     
	}
    }

Now we don't need to worry about our threads interfering with each other when they try to add to ``globalSum``, but we have a new problem. The program does not wait for the ``begin`` statements to finish running, so it will print ``globalSum`` before the algorithm has had enough time for each thread to add to it.

This displays another instance of a race condition, only instead of threads repeatedly trying to update the same global variable as the ``forall`` algorithm did, this algorithm has each thread only access the global variable once but doesn't wait for each thread to finish before it moves on. You can see this happen if you run the proceedure with the ``feedback`` flag set to true, and change the value of ``numRect`` to see how it affects the results.

 .. code-block:: java

	./pi --method=beginRace --feedback=true --numRect=1000

We can easily solve this race condition by using a ``sync`` statement, which will block the program from moving forward until a particular chunk of code has finished all of the ``begin`` statements it has started. In this case, we do not need to change anything within the ``areaBegin(init)`` proceedure itself, we just need to add a ``sync`` statement to the for loop that calls the proceedure with ``begin`` statements:

  .. code-block:: java

	if (method == "begin") {
	    sync for i in 1..numThreads { 
	      // The sync statements ensures that tasks created in the for loop
	      // must complete before continuing past the sync statement.     
	        begin areaBegin(i);     
	    }
	}

Run the code with the method flag set to ``begin`` and see how long it usually takes. Compare this with the time that ``linear`` usually takes. Change the value of ``numRects`` for both of these and see how the two algorithms compare with larger and smaller values of ``numRects``. Why would the algorithms differ in relative effeciency for different workloads?


``reduce`` Operator
*******************

The ``reduce`` operator can combine or compare values of operations that it can assign to run on different threads. In this first example of using ``reduce`` to solve the problem, we create the variable ``partials``, which we assign as a ``domain``, a dynamically resizable range of indicies. The ``reduce`` operator called on ``globalSum`` runs the proceedure ``rectArea`` for each index in ``partials`` (which we set to size ``numRect``) and adds all the results together.

 .. code-block:: java

    proc rectArea(i : int) { 

       // Calculates the area for one rectangle, so that a reduce operator
       // can calculate areas for all the rectangles in a domain.

         var width: real = 2.0 / numRect;
	 var x: real = -1 - width/2 + i*width;
	 return width * sqrt(1.0-x*x);         
    }

    if (method == "reduceRect") {
	 var partials: domain(1) = (1..numRect);
	 globalSum = + reduce (rectArea(partials));
    }

We can also use the ``reduce`` statement to split the workload up by threads rather than by rectangles in the circle. Notice that this algorithm looks very similar to the ``begin`` algorithm, because they both calulculate area for 1/``numThreads``'s worth of the rectangles, only they add them together through different means.


 .. code-block:: java

    proc areaReduceThread(init : int) { 

      // Calculates the area under the curve for 1/numThreads worth 
      // of rectangles and returns a partial sum

	 var width: real = 2.0 / numRect;
	 var partialSum: real = 0.0;
	 var x: real = -1 + width/2;
	 var i: int = init;
	 do {
	     x = -1 + ( i + 0.5) * width;
	     partialSum += sqrt(1.0 - x*x) * width;	    
	     i += numThreads;
	 }  while (i < numRect-1);
	 if (feedback) { writeln(partialSum);} 
	 return partialSum;         
    }

    if (method == "reduceThread") {
      var threads: domain(1) = (1..numThreads);
      globalSum = + reduce (areaReduceThread(threads));   
    }

Does one of the algorithms for reducing the problem work more efficiently, and why do you think it would work that way?
