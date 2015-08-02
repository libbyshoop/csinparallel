********************
Queue Exercise Hint
********************

Did you get your code working? The exercise may feel challenging at
first because the context is new. Let’s try to organize our thoughts a
bit with some pseudocode:

::

    '''
    def greet2():
        for 5 times
            get name from queue
            say hello

    def sendName2():
        queue
        make a child process, give it the queue
        start it

        for 5 times
            sleep for a bit
            put another name in the queue
    '''



When we work in psuedocode, it frees us from having to think about new
syntax and other details all at once. Instead we’re free to get some big
picture ideas down first. In the above psuedocode you can see that
the parent process, in ``sendName2``, will make a queue and a child process.
It will then loop five times, sending one piece of data at a time to the
child via the queue. The child, in ``greet2``, will also loop five times,
getting something from the queue and printing. Recall that if the child
attempts to get something from the queue when there’s nothing there, it
will block until something is available to get. If you didn’t already
solve this problem, try again now, using the pseudocode as a guide.

With this pseudocode developed, the actual code comes much more easily.
