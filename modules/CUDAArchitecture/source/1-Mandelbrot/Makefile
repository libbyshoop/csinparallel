CC 			= gcc
CFLAGS 		= -Wall -g
LFLAGS 		= -lX11 -lgomp -lm
XFLAG 		= -D SHOW_X
BFLAG 		= -D BENCHMARK
NVCC 		= nvcc
CUDA_FLAGS 	= -gencode arch=compute_20,code=sm_20 -g

mandelbrot : mandelbrot.cu
	$(NVCC) $(CUDA_FLAGS) $(XFLAG) mandelbrot.cu -o mandelbrot $(LFLAGS)

benchmark : mandelbrot.cu
	$(NVCC) $(CUDA_FLAGS) $(BFLAG) mandelbrot.cu -o benchmark $(LFLAGS)

XBenchmark : mandelbrot.cu
	$(NVCC) $(CUDA_FLAGS) $(BFLAG) $(XFLAG) mandelbrot.cu -o XBenchmark $(LFLAGS)

all : mandelbrot XBenchmark benchmark

clean :
	rm -rf *.o mandelbrot XBenchmark benchmark
