#include <omp.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
using namespace std;

/* Demo program for OpenMP: computes trapezoidal approximation to an integral*/

const double pi = 3.141592653589793238462643383079;

int main(int argc, char** argv) {
  /* Variables */
  double a = 0.0, b = pi;  /* limits of integration */;
  int n = 1048576; /* number of subdivisions = 2^20 */
  double integral; /* accumulates answer */
  int threadct = 1;  /* number of threads to use */
  
  /* parse command-line arg for number of threads, n */
  if (argc == 2) {
    threadct = atoi(argv[1]);
  } else if (argc == 3) {
    threadct = atoi(argv[1]);
    n = atoi(argv[2]);
  }
  double h = (b - a) / n; /* width of subdivision */

  double f(double x);
    
#ifdef _OPENMP
  cout << "OMP defined, threadct = " << threadct << endl;
#else
  cout << "OMP not defined" << endl;
#endif

  integral = (f(a) + f(b))/2.0;
  int i;

double start = omp_get_wtime();

#pragma omp parallel for num_threads(threadct) \
     shared (a, n, h) reduction(+: integral) private(i)
  for(i = 1; i < n; i++) {
    integral += f(a+i*h);
  }
  
  integral = integral * h;

// Measuring the elapsed time
double end = omp_get_wtime();
// Time calculation (in seconds)
double time1 = end - start;

  cout << "With n = " << n << " trapezoids, our estimate of the integral" <<
    " from " << a << " to " << b << " is " << integral << endl;
  cout << "Time for paralel computation section: "<< time1 << "  milliseconds." << endl;
}
   
double f(double x) {
  return sin(x);
}
