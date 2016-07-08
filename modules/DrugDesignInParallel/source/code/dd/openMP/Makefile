CXX=g++

LDFLAGS=-lm
OPENMP_FLAGS=-fopenmp

TBB_LIBS = -ltbb -lrt

dd_omp:  dd_omp.cpp
	$(CXX) -o dd_omp dd_omp.cpp $(LDFLAGS) $(OPENMP_FLAGS) $(TBB_LIBS) 

clean:
	rm dd_omp
