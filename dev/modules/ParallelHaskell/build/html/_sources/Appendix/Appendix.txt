Appendix
==========

Introductory Parallel Haskell Assignment
********************************************

This could be assigned after a class has reviewed the parallel tools ``par`` and ``pseq``. Students should have been taught a Haskell quicksort or similar program. The following code, the classic quicksort algorithm, is a refresher::

	1   quicksort :: (Ord a) => [a] -> [a]
	2   quicksort [ ] = [ ]
	3   quicksort (x:xs) = lesser ++ x:greater
	4        where   lesser = quicksort [y | y <- xs, y < x]
	5                greater = quicksort [z | z <-xs, z >= x]

1. **Briefly explain how this code functions; Where is the recursive step? Why is the line** ``quicksort [ ] = [ ]`` **needed?**

The code is a recursive quicksort. The line ``quicksort [ ] = [ ]`` is the base case, in order to prevent the recursion from going on forever. The next line, ``lesser ++ x:greater`` takes the list returned from ``lesser``, and concatenates it with ``x:greater``. ``x:greater`` appends ``x`` to the list returned from ``greater``. ``lesser`` is the first recursive call. It quicksorts all data valued less than ``x``. ``greater`` is the second recursive call. It quicksorts all data valued greater than ``x``.

2. **We want to end up parallelizing this code. What parts of the program can be run in parallel?**

Each quicksort call makes two recursive calls with independent sublists: ``lesser`` and ``greater``. ``lesser`` can be evaluated in parallel with ``greater``, at any level of the recursion because they do not share any data, and they can work independently of each other.

3. **We now want to include the** ``par`` **function, which evaluates its left argument in parallel while moving on to its right argument. We can start by rewriting the third line of code as,** ``quicksort (x:xs) = lesser `par` (lesser ++ x:greater)`` **. Would this successfully sort the data in parallel? Why or why not?**

As it is, this line of code will not work. Remember that the right side of ``par`` evaluates at the same time as the left side. However, here, the right side of ``par`` concatenates ``lesser`` with ``x:greater``. But the left side of ``par`` is still evaluating ``lesser``. How can we concatenate the list returned from lesser with another list if lesser is not yet done evaluating? This code runs the risk of returning an incomplete or unsorted list.

4. **How would we add the** ``pseq`` **function in order to successful complement the** ``par``  **function and make the program run in parallel (hint: the placement of** ``par`` **in the previous function is correct; it just needs a** ``pseq`` **to supplement it). Explain your reasoning and what happens in the code execution)?**

The line should look like this: ``quicksort (x:xs) = lesser `par` (greater `pseq` lesser ++ x:greater)``. The code now successfully operates in parallel. The ``par`` function tells Haskell to begin evaluating ``lesser`` while moving on to the right side of its argument, which in this case is the code in the parentheses. So Haskell is evaluating ``lesser`` while moving on to ``(greater `pseq` lesser ++ x:greater)``. ``pseq`` on the other hand, is doing the opposite. It begins evaluating its left argument, ``greater``, and refuses to let the code execute further until all arguments to its left are completed. This means that ``pseq`` guarantees that the ``lesser ++ x:greater`` concatenation cannot occur until both ``lesser`` and ``greater`` have been completely evaluated, thereby ensuring that the code comes back from its recursive calls correctly.