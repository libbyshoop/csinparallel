#include <iostream>
#include <cmath>
using namespace std;

#include "tbb/tbb.h"
using namespace tbb;

/* Demo program for TBB: computes trapezoidal approximation to an integral*/

const double pi = 3.141592653589793238462643383079;

double f(double x);
     
class SumHeights {
  double const my_a;
  double const my_h;
  double &my_int;

public:
  void operator() (const blocked_range<size_t>& r) const {
    for(size_t i = r.begin(); i != r.end(); i++) {
      my_int += f(my_a+i*my_h);
    }
  }
  
  SumHeights(const double a, const double h, double &integral) : 
    my_a(a), my_h(h), my_int(integral) 
  {}
};

int main(int argc, char** argv) {
   /* Variables */
   double a = 0.0, b = pi;  /* limits of integration */;
   int n = 1048576; /* number of subdivisions = 2^20 */

   double h = (b - a) / n; /* width of subdivision */
   double integral; /* accumulates answer */
   
   integral = (f(a) + f(b))/2.0;

   parallel_for(blocked_range<size_t>(1, n), SumHeights(a, h, integral));
   
   integral = integral * h;
   cout << "With n = " << n << " trapezoids, our estimate of the integral" <<
     " from " << a << " to " << b << " is " << integral << endl;
}
    
double f(double x) {

   return sin(x);
}
