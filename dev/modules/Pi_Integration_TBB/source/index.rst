.. Pi Using Numerical Integration: TBB documentation master file, created by
   sphinx-quickstart on Wed Jun 05 11:14:11 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pi Using Numerical Integration: TBB
===============================================================


There are two major parts to the TBB solution. The first, after the required #include lines to import the TBB definitions, is the call to parallel_reduce(). This algorithm call will take a range (0, num_rect), and a body object (area), and a partitioner as parameters. The range will be divided into sub-ranges until a sub-range is deemed small enough, which is the function of the partitioner. This range will be encapsulated into a task that can be executed by a thread.

Once the computation is complete, the sum of all the rectangle areas computed for the smallest sub-ranges has been gathered (reduced) into the sum component of the area object. Multiply this value by 2.0 to compute the approximation of pi. ::

 #include "tbb/parallel_reduce.h"
 #include "tbb/task_scheduler_init.h"
 #include "tbb/blocked_range.h"
 
 using namespace std;
 using namespace tbb;
 long long num_rect =  1000000000;
 . . .
 double pi;
 double width = 1./(double)num_rect;
 MyPi area(&width);  //construct MyPi with initializer of step(&width)
 parallel_reduce(blocked_range<size_t>(0,num_rect),
                area,
                auto_partitioner());
 pi = area.sum * 2.0;

The second major part of the solution is the body class MyPi defined below. This class defines the operator() method with the body of the serial code loop. Once a task has been defined (sub-range has been deemed indivisible), the loop in the operator() method computes the midpoint of the associated rectangle for each value within the range of the task. From this, the area of that rectangle is computed and added to the object’s sum component.

Once the entire range within a task has been used, the sum components from different tasks are added together through the join() method.  This method accepts the sum from some other task and adds it to the local sum of the current task. This sum can then be used in another join() operation until the final sum of all tasks has been added together. This final result is then available through the sum component of the original body object used in the parallel_reduce() call. ::

 class MyPi {
 double *const my_h;
 
 public:
 double sum;
 
 void operator()( const blocked_range<size_t>& r ) {
 double h = *my_h;
 double x;
      for (size_t i = r.begin(); i != r.end(); ++i){
    	x = -1 + (i + 0.5) * h;
    	sum = sum + sqrt(1.0 - x*x) * h;
  	}
 }
 void join( const MyPi& y ) {sum += y.sum;}
 
 MyPi(double *const step) : my_h(h), sum(0) {}
 MyPi( MyPi& x, split ) : my_h(x.my_h), sum(0) {}
 };
