/*
// Copyright 2013 Intel Corporation. All Rights Reserved.
// 
// The source code contained or described herein and all documents related 
// to the source code ("Material") are owned by Intel Corporation or its
// suppliers or licensors.  Title to the Material remains with Intel Corporation
// or its suppliers and licensors.  The Material is protected by worldwide
// copyright and trade secret laws and treaty provisions.  No part of the
// Material may be used, copied, reproduced, modified, published, uploaded,
// posted, transmitted, distributed, or disclosed in any way without Intel's
// prior express written permission.
// 
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise.  Any license under such intellectual property rights must
// be express and approved by Intel in writing.
*/

#include "fib_cilk_mic.h"

#define SIZE 20

int fib_serial (long long int N)
{
    long long int x, y;
    if (N < 2)
        return N;

    x = fib_serial(N - 1);
    y = fib_serial(N - 2);
    return x + y;
}

int fib(long long int N)
{
    long long int x, y;

    if (N < 15) return fib_serial(N); 
    x = cilk_spawn fib(N - 1);  // non-blocking . x & y values
    y = fib(N - 2);             // calculated in parallel
    cilk_sync;                  // wait here until both done
    return x + y;
}


float compute(long int N)
{
    std::cout << "Fibonacci #" << N << ":" << fib(N) << std::endl;
    return (0.);
}

int prepare(long int Count)
{
    int i, j, n = Count;
    return (0);
}

int cleanup(long int N)
{
    return (0);
}

int main(int argc, char *argv[])
{
    long int Count = SIZE;
    int Error;

    if (argc > 1)
    {
        Count = std::atoi(argv[1]);
        if (Count <= 0)
        {
            std::cerr << "Invalid argument" << std::endl;
            std::cerr << "Usage: " << argv[0] << "N" << std::endl;
            std::cerr << "       N = size" << std::endl;
            return 1;
        }
    }

    std::cout << "counts:" << Count << std::endl;
    std::cout << "preparation starting" << std::endl;
    if (Error = prepare(Count) != 0)
        return Error;
    std::cout << "preparation done" << std::endl;
    unsigned long long start_ticks = my_getticks();
    Error = compute(Count);
    unsigned long long end_ticks = my_getticks();
    unsigned long long ticks = end_ticks - start_ticks;

    if (Error == 0)
        std::cout << "succeeded in ";
    else
        std::cout << "failed in ";
    std::cout << my_ticks_to_seconds(ticks) << " seconds." << std::endl;
    std::cout << "starting cleanup" << std::endl;
    return cleanup(Count);
}
