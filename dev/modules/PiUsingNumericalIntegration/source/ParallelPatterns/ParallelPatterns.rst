=================================
Looking Ahead: Parallel Patterns
=================================

Parallel Design Patterns  
-------------------------

As described in the module Introduction to Parallel Design Patterns, the OPL system of parallel patterns is organized into four levels. 
  * The top level, called the **Application Architecture Level**, consists of structural and computational patterns for designing large pieces of software (whether or not that software will be implemented with parallelism). 
  * Implementing this problem will not require a large software application – it’s more on the scale of a particular iterative computation *within* an application. Therefore, we will not need to consider a combination of structural and computational patterns to solve this problem. We need only find a way to add up many areas of rectangles, then multiply by 2. 
  * The next level of OPL patterns is the **Parallel Algorithms Strategy Level**. Two patterns at this level relate to our discussion above.
    - The **Data parallel** pattern involves applying the same computational operations to multiple data values, assuming that those operations can be performed independently for different data values. We can consider the values xias our multiple data values, and the rectangle-area computation as our operations.
    - The **Geometric decomposition pattern** involves breaking a large computation up into smaller “chunks” that we can compute in parallel. If we use a large number of rectangles N in our area approximation, we can apply this pattern by having multiple parallel computations that each add the areas of a subset of the rectangles.

**Note**: These patterns (as we have just described them) indicate strategies for designing parallel solutions to our problem which will prove useful when we write parallel code to compute π. However, our problem may not quite fit the exact OPL definitions of these terms. In particular, OPL refers to a data structure for both of these pattern names, whereas we might simply add up areas of rectangles in a loop without an explicit data 
structure of values. Also, the OPL **Geometric decomposition pattern** refers to concurrently updatable “chunks,” whereas our “chunks” of data values will never change. Nevertheless, these two algorithmic strategy patterns will help guide us to good parallelizations for given computational platforms.
