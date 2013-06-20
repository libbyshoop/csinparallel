********************************************
Parallelizing with Go: Sequential Code Given
********************************************

Simulating in Go
################

Structure
---------

Go is a language developed to combine features of C/C++ and Python with intuitive concurrency.

Go code for simulating this epidemic is available at :download:`epidemic.go <code/epidemic.go>`. Read through the code and the comments, then follow the directions below to take advantage of the simplicity of parallelizing in Go.


Parallelizing
-------------

- Go provides simple options for parallelization using goroutines. Briefly, a goroutine is a function that can run simultaneously, or concurrently, with other sections of code. Goroutines can be thought of similarly to threads, although they are not exactly synonymous. A Go program may use thousands of goroutines since they are very lightweight and are managed behind the scenes. If a programmer writes code thinking of goroutines as threads considering race conditions and deadlock, they will most likely be successful

- When using goroutines, it is important to be able to communicate between goroutines. This is accomplished by channels which can be thought of as conveyor belts. Information is put in the channel by one goroutine and taken out by another. They can also be compared to work queues since information can be added to and removed from by multiple parties. Channels can convey any data type of information including structs and are also used buffered or unbuffered.

- The Sync library provides an easy method to ensure concurrency and timing of goroutines. This package defines the type WaitGroup which manages the threads that should be executed loosely as a group. This is accomplished by a few methods defined for a WaitGroup such as add() and done(). As goroutines are created, add() should be called which will increment a counter (of goroutines) within the WaitGroup. When a goroutine has finished executing, call Done() which will decrement the counter. There is a blocking method called Wait() that will block until the counter reaches 0 signifying that all threads have finished executing. Other options include Locks which mimic a physical lock that can be open and closed by goroutines, but we did not pursue these for our simulation

- To parallelize the go simulation we used several (in fact, hundreds) goroutines to divide up the computation of the simulation. The parallelization can be likened to parallelizing loops with OpenMP. Within a for loop, create a new goroutine to execute the body of the loop for each iteration. The body of the loop will need to be defined in a function and the appropriate variables pass (by reference if needed).

- To communicate and share information between goroutines, create a channel used to share and store infected people. As goroutines identify people as infected, they add the person to the channel. Different goroutines should later remove people from the channel and compare the distance between all susceptible people and this infected individual. 

- Unfortunately if a channel is empty for too long, it will close. To ensure all infected people are added to the channel a blocking mechanism is necesary. Use a WaitGroup and add all goroutines that are involved in searching for infected persons. Call Wait() in the WaitGroup before beginning to remove people from the channel. When the last of these goroutines have finished their search and have called Done(), the call to Wait() will finish blocking and the program will continue execution.
