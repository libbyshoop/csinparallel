**********
Background
**********

#########
History
#########

* DFAs, NFAs, pushdown automata, Turing machines... All are mathematical entities that model computation. These abstract systems have concrete, practical applications in computer science (CS).

  For example, deterministic finite automata (DFAs) are associated with regular expressions, which computer   programs that involve pattern matching frequently rely on. Also, knowing theoretical results such as the inability of any computation to determine whether or not another computation will stop (Halting Problem) can keeps programmers from attempting to write impossible computer programs.

* Automata represent one approach to mathematically modeling computation. There are others.

  For example, the mathematical logician Alonzo Church created a formalism of computation based on functions in the 1930s, called the :math:`\lambda`*-calculus*. The key notion in this approach is an operator (i.e., function) called :math:`\lambda` that is capable of generating other functions.

  One of the earliest high-level programming languages, LISP (for LISt Processing language, 1959), is a practical computer implementation of the :math:`\lambda`-calculus. LISP was designed originally for research in *artificial intelligence (AI)*, a field in CS that perpetually seeks to extend the capabilities of computers to carry out tasks that humans can do. Scheme and Clojure are some contemporary programming languages descended from the original LISP, and and other widely used "functional" programming languages such as ML and Haskell are based on the :math:`\lambda`-calculus. Programmers use these languages to develop useful applications, and researchers use them to explore new frontiers in computing.

* From a theoretical viewpoint, the :math:`\lambda`-calculus embodies all essential features of functional computation. This holds because the relationship between "inputs" (domain values in Mathematics, arguments/parameters in programming) and "outputs" (range values in Math, return values in programming) from functions expresses everything in a purely functional system of computations (no "state changes"), and :math:`\lambda`-calculus is the mathematical theory of functions considered entirely according to their "inputs" and "outputs."

  In fact, it can be proven that any other foundation for *functional* computation, such as Turing machines (which can express any type of computation), will have exactly the same expressive power for functional computation as the :math:`\lambda`-calculus [Pierce 95].

* However, all of the computational models we've mentioned so far (Turing machines, :math:`\lambda`-calculus, etc.) are for *sequential computations* only. This means that we assume only a single computational entity. Until a few years ago, it was reasonable to assume that only one computational processor would be available for most computations, because most computers had only one computational circuit for carrying out instructions.

#######################
The Push to Parallelism
#######################


* Nowadays, retailers sell only *multi-core* computers (i.e., computers having multiple circuits for carrying out instructions) on the commodity market, and hardware manufacturers such as Intel and AMD no longer produce chips with only one computational processor. This results from computer engineering having reached certain limitations on performance for individual processors (related to electrical power consumption, access to computer memory, and parallel speedup capabilities with a single processor).

* Consequently, the only way to continue improving the performance of computers going forward is to use *parallel computing*, in which multiple computer actions are carried out physically at the same time. Parallel computing (or *parallelism*) can be accomplished by writing programs that use multiple computational cores at the same time, and/or by running multiple cooperating programs on multiple computers.

* Some computations are easy to parallelize. For example, a computation may involve applying exactly the same program steps to multiple independent input data sets, in which case we can perform parallel processing by executing that series of program steps on multiple processors (i.e., multiple cores and/or computers), and submitting different data sets to different processors. We call this strategy *data parallelism*. Some authors refer to such computations as being *embarassingly parallel*.

* Other types of computations may be parallelizable without being data parallelizable. For example, matrix multiplication requires combining the rows and columns of rectangular arrays of numbers in ways that require accessing each number multiple times, in different groupings. Parallelization strategies for matrix multiplication exist, such as multiplying submatrices formed by subdividing the original matrices, then combining those results appropriately. However, those strategies are more complex than simple data parallelism.

#################################
Difficulties with Parallelization
#################################

* Many computations require parallelizing according to the computational steps instead of (or in addition to) parallelizing according to the data. When a computation has multiple processors carrying out different sequences of computational steps in order to accomplish its work, we say that computation has *task parallelism*.

  For example, imagine a computation that extracts certain elements from a body of text (e.g., proper names), then sorts those elements, and finally removing duplications. With multiple processors, one might program one processor to extract those elements, another to perform the sorting operation, and a third to remove the duplications. In effect, we have an assembly line of processes, also called a *pipeline* by computer scientists.

* Computer scientists have found other computations exceedingly difficult to parallelize effectively. Notably, nobody knows how to parallelize finite state machines (FSMs) well, as a general class of computations. [View from Berkeley 06, p.16]

##################
The Solution
##################


* We can easily imagine how to construct a mathematical model of computation for simple data parallelism from a model of computation for the sequential case of that same computation, by replicating the sequential model. This approach seems promising as long as we can assume that those multiple parallel computations do not need to interact with each other in any way.

* However, more complicated forms of parallelism that involve multiple processes interacting in various ways, such as the task parallelism example of pipelining, requires a mathematical model of parallel computation capable of expressing those interactions between processes.

The :math:`\pi`-calculus, introduced in the next section, is an example of such a model of parallel computation.

