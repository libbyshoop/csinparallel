.. Go Language documentation master file, created by
   sphinx-quickstart on Wed Jun 05 10:05:08 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pi Using Numerical Integration: Go Language
============================================

.. _here: https://code.google.com/p/eapf-tech-pack-practicum/source/browse/trunk/pi_integration/pi_area_go.go

Go is a new open-source language with built-in support for concurrency developed at Google. You may get more info on the language at http://golang.org/. In particular, here is an introduction into concurrency constructs: http://golang.org/doc/effective_go.html#concurrency.  You can find the program in easily downloadable form here_. ::


	package main

	import (
		"flag"
		"fmt"
		"math"
		"runtime"
	)

	var (
		nRect  = flag.Int("rect", 1e6, "number of rectangles")
		nGrain = flag.Int("grain", 1e4, "parallel task grain size (in rectangles)")
		nCPU   = flag.Int("cpu", 1, "number of CPUs to use")
	)

	func main() {
		flag.Parse()
		runtime.GOMAXPROCS(*nCPU)   // Set number of OS threads to use.
		nParts := 0                 // Number of parallel tasks.
		parts := make(chan float64) // Channel to collect part results.
		for i := 0; i < *nRect; i += *nGrain {
			nParts += 1
			end := i + *nGrain // Calculate end of this part.
			if end > *nRect {
				end = *nRect
			}
			// Launch a concurrent goroutine to process this part.
			go func(begin, end int) {
				sum := 0.0
				h := 2.0 / float64(*nRect)
				for i := begin; i < end; i++ {
					x := -1 + (float64(i)+0.5)*h
					sum += math.Sqrt(1-x*x) * h
				}
				parts <- sum // Send the result back.
			}(i, end)
		}
	
		// Aggregate part results.
		sum := 0.0
		for p := 0; p < nParts; p++ {
			sum += <-parts
		}
		pi := sum * 2.0
		fmt.Printf("PI = %g\n", pi)
	}	
	 	
