****************************************
Solution to Exercise
****************************************

A solution is to the hole-digging exercise is as follows:

::

    def dig(workerName, holeID, lock):
        lock.acquire()
        print "Hiddy-ho!  I'm worker", workerName, "and today I have to dig hole", holeID
        lock.release()

    def assignDiggers():
        lock = Lock()
        workerNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        for holeID in range(len(workerNames)):
            Process(target=dig, args=(workerNames[holeID], holeID, lock)).start()



The ``assignDiggers`` function
creates a single Lock object and a list of the worker names. A process
is started for each worker, passing the appropriate name, assigned hole,
and the lock. Each worker attempts to acquire the lock, and is only
allowed to print once that acquisition succeeds. After printing, the
worker releases the lock so that another worker may print.

This exercise is also a good demonstration of the strengths and
limitations of different approaches to looping. The solution shown above uses what can be referred to as a
“loop-by-index” approach, in which the holeID index is the loop
variable. An alternative would be a “loop-by-element” (for-each loop)
approach, like this:

::

        ... # Other code as before
        for workerName in workerNames:
            Process(target=dig, args=(workerName, workerNames.index(workerName), lock)).start()

The loop-by-element
approach, however, is not as effective, because the
worker-Names.index(workerName) requires a fair amount of extra work to
execute. While the actual execution time will be nearly instantaneous in
both approaches for a small list, it is nevertheless a good idea to reiterate the general
principle of using the right programming constructs for maximum
efficiency.  You don't want to fall into the trap of using a less efficient choice on a larger list of data in some other circumstance, or in a circumstance where you might execute such a loop over and over many times, where the time used will add up.


