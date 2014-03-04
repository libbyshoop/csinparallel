/* critical3.cpp
 * ... compares the use of OpenMP's critical and atomic directives,
 *  providing an example of when critical works but atomic does not.
 *
 * Joel Adams, Calvin College, February 2014.
 *
 * Usage: ./critical3
 *
 * Exercise:
 *  - Compile, run, note behavior.
 *  - Uncomment #pragma on line A, note behavior as threads 'race'
 *     to insert values into cout (a shared output stream).
 *  - Uncomment #pragma on line B, try to compile, note error message.
 *  - Recomment #pragma on line B; uncomment #pragma on line C,
 *     recompile, rerun, note results.
 */

#include <iostream>      // cout, <<, etc.
#include<omp.h>          // openMP 
using namespace std;

int main() {
    cout << "\nBefore ...\n" << endl;

//    #pragma omp parallel                        // A
    {
        int id = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
//        #pragma omp atomic                      // B
//        #pragma omp critical                    // C
        cout << "Hello from thread #" << id 
             << " out of " << numThreads << endl;
    }

    cout << "\nAfter ...\n" << endl;
}

