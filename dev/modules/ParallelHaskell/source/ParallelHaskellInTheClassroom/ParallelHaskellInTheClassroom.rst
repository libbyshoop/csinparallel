Parallel Haskell in the Classroom
====================================

The parallel programming tools available in Haskell can be easily used. However, while functional programming is compelling and powerful, it is also very different from other programming languages with which most undergraduates are familiar. It may be difficult to teach Haskell, and this could distract students from the focus of learning concurrent and parallel programming practices. One solution to this problem is to introduce this module in a classroom where students have a background in functional programming, like a programming languages course. If students are already familiar with functional programming, they can be focused on learning the parallel part of the code, deriving the maximum educational benefit from the work shown here. 

Many programming languages textbooks introduce functional programming and parallel computing \ :sup:`[10, 14]`\. Instructors could supplement their reading with the free *Real World Haskell* \ :sup:`[12]` and *Learn You a Haskell for Great Good* \ :sup:`[5]`\. We especially recommend the latter. Lipovaça provides a complete, entertaining background to functional coding that is especially accessible to young programmers. Before being introduced to the parallel tools in Haskell, we recommend that students go over chapters 2-6 and chapter 9 in *Learn You a Haskell*. This should provide them with a sufficient background in recursion, higher-order functions, and IO for parallelization in Haskell.


Discussion
--------------

As parallel computing becomes increasingly ubiquitous in the programming world, it has become important that parallel programming be taught at the undergraduate level. To that end, there exist many parallel languages and environments, and educators must carefully choose which ones to introduce to young programmers. We believe that the parallel constructs in Haskell are both powerful and informative about the nature of parallel programming and coordination. 

Haskell can be brought to the undergraduate classroom by assigning small, simple programs, like our Riemann estimation of π, or a parallel quicksort. The advantage to these simple assignments is that the volume of work is fairly small. Successfully implementing the π estimation in parallel, for example, should take no more than a dozen lines of code. However, the difficulty of understanding the language constructs and using their syntax makes the assignment far from trivial. A challenging problem solved with relatively few lines of code will compel students to experiment with Haskell's various parallel tools before turning in the assignment. This is ideal, because such exploration will allow students to gain a more holistic understanding of the language in general, including how Haskell interacts with multiple CPUs. 

The next step is to begin investigating the potential benefit of more long-term parallel assignments. There are many complex, "embarrassingly parallel" programs that could be implemented with Haskell. The way to introduce this more advanced work to undergraduates has yet to be explored. Certainly, a long term project in which students are required to implement a complicated system, like the Boids simulation \ :sup:`[7]`\  or Conway's Game of Life, would increase the requisite level of Haskell comprehension. At the same time, such a project may not actually provide insight into the parallel functions of Haskell beyond what is learned in our small examples; the project would probably make students better programmers, but it is not clear if it would make them better parallel programmers. Furthermore, the advantage to the short assignment structure that we have introduced here is that it need not only be used in a course dedicated to parallel computing. The brevity of the program means that a module could be covered in a programming languages course over the span of about a week, perhaps right as the course is covering functional languages.

Acknowledgements
*******************

Partial support for this work was provided by the National Science Foundation, by the John S. Rogers
Science Research Program, and by the James F. and Marion L. Miller Foundation. We would also like to
thank David Bunde, Peter Drake, and James Duncan.

References 
************

[1] Joel Adams, Richard Brown, and Elizabeth Shoop. Patterns and exemplars: Compelling strategies for teaching parallel and distributed computing to CS undergraduates. In *EduPar-13*, 2013.

[2] Don Jones, Jr., Simon Marlow, and Satnam Singh. Parallel performance tuning for Haskell. In *Proceedings of the 2nd ACM SIGPLAN symposium on Haskell*, Haskell '09, 2009.

[3] Simon Peyton Jones and Simon Marlow. The glorious glasgow haskell compilation system user's guide, version 7.6.2. http://www.haskell.org/ghc/docs/7.6.2/html/users_guide/. Accessed: 2013-06-10.

[4] Simon Peyton Jones and Satnam Singh. A tutorial on parallel and concurrent programming in haskell. In *Proceedings of the 6th international conference on Advanced functional programming*, 2009.

[5] Miran Lipovaça. *Learn You a Haskell for Great Good!* No Starch Press, San Francisco, CA, 2011.

[6] Simon Marlow. *Parallel and Concurrent Programming in Haskell*. O'Reilly & Associates Inc, Sebastopol, CA, 2013.

[7] Craig Reynolds. Flocks, herds and schools: A distributed behavioral model. In *Proceedings of the 14th annual conference on Computer graphics and interactive techniques*, 1987.

[8] Wilson Rivera. How to introduce parallelism into programming languages courses. In *EduPar-13*, 2013.

[9] Suzanne Rivoire. A breadth-first course in multicore and manycore programming. In *Proceedings of the 41st ACM technical symposium on Computer science education*, 2010.

[10] Michael Scott. *Programming Language Pragmatics*. Elsevier/Morgan Kaufmann Publishers, Amsterdam, 2009.

[11] Paul Steinberg and Matthew Wolf. ACM Parallel Computing TechPack. http://techpack.acm.org/parallel/. Accessed: 2013-05-30.

[12] Bryan Sullivan. *Real World Haskell*. O'Reilly & Associates Inc, Sebastopol, CA, 2009.

[13] Prabhat Totoo, Pantazis Deligiannis, and Hans-Wolfgang Loidl. Haskell vs. F# vs. Scala: A high-level language features and parallelism support comparison. In *Proceedings of the 1st ACM SIGPLAN workshop on Functional high-performance computing*, 2012.

[14] Allen Tucker. *Programming languages: Principles and paradigms*. McGraw-Hill Higher Education, Boston, 2007.