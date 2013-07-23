.. Pi Using Numerical Integration: Haskell documentation master file, created by
   sphinx-quickstart on Wed Jun 05 11:44:03 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pi Using Numerical Integration: Haskell
===================================================================


Haskell is a functional programming language that was created in the 1980's. It is a static, strongly-typed language which incorporates automatic type reference. It also has built-in parallel interfaces that can make it easier to implement parallel programming. To show the Haskell implementation, we will compare the sequential implementation and the data-parallel implementation.

Sequential Riemann
-------------------

The program will take an argument for the number of partitions and return
an estimation of pi. It will do this by the method of right-handed Riemann
rectangle summation. To implement this sum we do the following. First we
create a list that has an appropriate dx based on the number of partitions the
user inputs. We then multiply dx by twice the height of the right hand point to
get the area of our rectangle. We then add up all of the area of the rectangles
to get our approximation. ::


 -- Equation for the upper hemisphere of the unit circle
 
 circle :: Double -> Double
 circle x = sqrt (abs(1 - x^2))
 
 -- Calculate the area of a right-handed Riemann rectangle
 
 area :: Double -> Double -> Double
 area x1 x2 = (x2 - x1) * circle x2

 -- Recursively add the areas of the Riemann rectangles

 estimate (x:[]) = 0
 estimate (x:y:xs) = (area x y) + estimate (y:xs)

Parallel Riemann
-------------------

The parallel version is almost identical code, using a similar recursive function
to add the areas of the Riemann rectangles. The primary difference comes from
the insertion of the **par** and **pseq** functions. In our parallel estimation of pi, **par**
is calculating *smaller*, and **pseq** is calculating *larger* at the same time. *larger*
makes a recursive call to **parEstimate**, giving us another smaller section to
begin executing in parallel. This essentially gives us a cascading sum of parallel
computations of the areas of the Riemann rectangles. Once *larger* --with the
recursive smallers-- is finally complete, larger and smaller are added together,
resulting in pi. ::

 import Control.Parallel

 -- Equation for the upper hemisphere of the unit circle

 circle :: Double -> Double
 circle x = sqrt (abs(1 - x^2))

 -- Calculate the area of a right-handed Riemann rectangle

 area :: Double -> Double -> Double
 area x1 x2 = (x2 - x1) * circle x2

 -- Recursively add the areas of the Riemann rectangles

 parEstimate :: [Double] -> Double
 parEstimate (x:[]) = 0
 parEstimate (x:y:[]) = area x y
 parEstimate (x:y:xs) =
 smaller `par` (larger `pseq` smaller + larger)
   where smaller = area x y
 larger = parEstimate (y:xs)

Further Exploration
--------------------
  * Try building and running both the sequential and the parallelized implementations of the Riemann sum in Haskell. Compare the timing results you collected for the sequential program to the time performance of this parallel program using various numbers of threads.  Does the parallel program perform better?  Is the speed up as much as you would expect?  If not, can you hypothesize why?
