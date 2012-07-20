#include <iostream>
#include <cmath>
using namespace std;

#include "tbb/tbb.h"
using namespace tbb;

/* Demo program for TBB: computes trapezoidal approximation to an integral*/

const double pi = 3.141592653589793238462643383079;

double f(double x);
     
class SumHeights2 {
  double const my_a;
  double const my_h;

public:
  double my_int;

  void operator() (const blocked_range<size_t>& r) {
    double a2 = my_a;
    double h2 = my_h;
    double int2 = my_int;
    size_t end = r.end();
    for(size_t i = r.begin(); i != end; i++) {
      int2 += f(a2+i*h2);
    }
    my_int = int2;
  }
  
  SumHeights2(const double a, const double h, const double integral) : 
    my_a(a), my_h(h), my_int(integral)
  {}

  SumHeights2(SumHeights2 &x, split) : 
    my_a(x.my_a), my_h(x.my_h), my_int(0)
  {}

  void join( const SumHeights2 &y) { my_int += y.my_int; }
};

int main(int argc, char** argv) {
   /* Variables */
   double a = 0.0, b = pi;  /* limits of integration */;
   int n = 1048576; /* number of subdivisions = 2^20 */

   double h = (b - a) / n; /* width of subdivision */
   double integral; /* accumulates answer */
   
   integral = (f(a) + f(b))/2.0;

   SumHeights2 sh2(a, h, integral);
   parallel_reduce(blocked_range<size_t>(1, n), sh2);
   integral += sh2.my_int;
   
   integral = integral * h;
   cout << "With n = " << n << " trapezoids, our estimate of the integral" <<
     " from " << a << " to " << b << " is " << integral << endl;
}
    
double f(double x) {

   return sin(x);
}
