Parallel Computing with Haskell using Quicksort
=================================================


When teaching parallel computing at the undergraduate level, the selection of a programming language is important. When choosing, there are two routes that an instructor can take \ :sup:`[4]`\. The first is to choose a programming language whose computation is parallelized through low-level additions. The alternative is to use a language whose computation can be parallelized through high-level abstraction. This module explores the latter route, which will increasingly lead to functional programming \ :sup:`[4]`\.

Students must be taught that computationally expensive problems can be divided into pieces and evaluated in parallel on multiple cores. The modern curriculum introduces parallel computing through the use of low-level programming languages and tools. Haskell is an interesting alternative because of its abstract parallel functions, which can make it much easier to program in parallel by removing many of the problems associated with lower-level parallelism \ :sup:`[6]`\.

We began our education in Haskell with the book, *Learn You a Haskell for Great Good*, by Miran Lipovaça
\ :sup:`[5]`\, and we supplemented this with selections from *Real World Haskell by Sullivan* \ :sup:`[12]`\. These publications,
along with other research conducted on the parallel language constructs available in Haskell, provided us
with a sufficient foundation in parallel programming.

Haskell is a purely functional, deterministic programming language. It has a static, strong, type system
with automatic type inference. The upshot is that the output of its functions will always be the same
regardless of whether it is running on one or multiple cores. This eliminates deadlock and race condition
errors that can plague traditional approaches to and parallel programming \ :sup:`[6]`\. Additionally, the abstraction
in the parallel functions in Haskell allows for the programs to run on a range of hardware \ :sup:`[6]`\. These
advantages lead us to ask the question: can a functional programming paradigm be an effective tool to teach
undergraduate students parallel computing?

To answer this question, instructors must weigh the benefits of Haskell against the overhead of teaching
undergraduate students a new programming language and the functional paradigm. Haskell has a variety
of contructs available for parallel programming. When imported, the ``Control.Concurrent`` module allows
the programmer to explicitly generate new threads. *Real World Haskell* defines a thread as "an IO action
that executes independently from other threads" \ :sup:`[12]`\. To create a new thread, we use the ``forkIO`` function.
We are not going to use the ``Control.Concurrent`` module in our lesson plan or assignment. We have found
that the concurrency module is ineffective in producing elegant parallel solutions to the problems we have
proposed. For more information about concurrent programming in Haskell, see *Real World Haskell*.

A higher level parallel solution is the ``Control.Parallel`` module, which allows the use of the ``par`` and
``pseq`` functions. With these two functions we are able to take a sequential program and easily change it
so that it can be evaluated in parallel. ``par`` evaluates its left argument while simultaneously executing the
right argument. ``pseq`` evaluates the expression on the left to `weak head normal form <http://www.haskell.org/haskellwiki/Weak_head_normal_form>`_ before returning the
one to the right. ``par`` and ``pseq`` are often used together to parallelize code. Here is an example of how we
would implement ``par`` and ``pseq`` into a quicksort algorithm. The first snippet of code is a simple sequential
quicksort algorithm, the second demonstrates where you would insert the ``par`` and ``pseq`` functions. ::

	1   import Control.Parallel
	2
	3
	4   -- A non-parallel quicksort
	5   quicksort :: Ord a => [a] -> [a]
	6   quicksort (x:xs) = smaller ++ x:larger
	7	 where
	8 	  	smaller = quicksort [y | y <- xs, y < x]
	9 		larger = quicksort [y | y <- xs, y >= x]
	10   quicksort _ = []
	11
	12   -- A parallel quicksort
	13   parQuicksort :: Ord a => [a] -> [a]
	14   parQuicksort (x:xs) = ($! smaller) `par` (($! smaller) `pseq`
	15   						(smaller ++ x:larger))
	16	where
	17 		smaller = parQuicksort [y | y <- xs, y < x]
	18 		larger = parQuicksort [y | y <- xs, y >= x]
	19   parQuicksort _ = []

GHC Compilation and Runtime Options
---------------------------------------

When compiling our parallel programs in Haskell, we experienced faster run times by including the following GHC run time options. By including the ``-rtsopts`` tag in our compilation statement we are able to include all of the runtime settings (``RTS``) features of Haskell. The ``-threaded`` tag, tells the compiler that the program will be run in parallel. Finally, the ``-O`` tag optimizes the program \ :sup:`[3]`\. All of these should be included when first compiling a program, in a statement similar to the following.

``ghc -threaded -rtsopts -O --make myProgram.hs``


Options can be passed to the GHC run time system. The ``+RTS`` command line option starts the run time options and ``-RTS`` ends them. Anything between these two tags are options for the run time system. All arguments after ``-RTS`` are arguments for the program. When we use the ``getArgs`` function from the ``System.Environment`` module to read the command line arguments, no run time options will be read \ :sup:`[12]`\. The performance of a parallel Haskell program can sometimes be improved by increasing the heap size of the program \ :sup:`[3]`\. This will reduce the number of garbage collections at run time. The stack and heap sizes are specified by the run time options. For example, ``-K100MB`` specifies a 100MB stack size and ``-H500MB`` specifies a 500MB heap size \ :sup:`[2]`\. In addition to increasing the size of the stack and heap, we disable parallel load-balance garbage collection with the ``-qb`` option. The ``-N`` option is used to specify the number of cores that the Haskell program should run on. The ``-s`` option is used to print useful run time statistics. The full command used to run the program is:

``./myProgram +RTS -N{number of cores} -qb -H{heap size} -k{stack size} -s -RTS``