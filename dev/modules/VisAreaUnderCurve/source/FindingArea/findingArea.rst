*******************************
Finding the Area Unde the Curve
*******************************

Integration or finding the area under a curve is a very important concept from calculus applied in life sciences, engineering and economics.

The basic concept behind finding the area under a curve is **approximating the area with rectangles**. 

Given two endpoints on the curve we can split this region into numerous small rectangles. The more rectangles we choose, the smaller they become, and the closer they approximate the curve. Therefore, choosing a larger amount of rectangles yields to more accurate results. Mathematically, the number of rectangles can approach infinity. In the code samples found here the number of rectangles specified may approach a large number. An exception to this would be if the calculation was to be visualized, as the width of a rectangle cannot be smaller than the pixel size of the screen used for display. 

There exist functions whose area we either know or can calculate exactly without integration. Such functions can provide a clear guideline to see how accurate our program is. Here we choose to examine the following functions:

* linear function *f* (*x*) = *x* in the positive range
* the unit circle in the range [0,1]
* *f* (*x*) = sin(*x*) in the range [0, pi ] 

The area of a linear function in the positive range is equivalent to the area of a triangle, which is 0.5 x height x width. 

The area of a quarter unit circle is quarter pi. 

Finally, the are of the sin function in the range from 0 to pi is exactly 2. 

It is important to keep these facts in mind as they will help us determine the correctness of our program. 
