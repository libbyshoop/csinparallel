CXX=g++

LDFLAGS=-lm
CPP11_THREADS = -std=c++11 -pthread
TBB_LIBS = -ltbb -lrt

dd_threads: dd_threads.cpp
	$(CXX) -o dd_threads dd_threads.cpp $(LDFLAGS) $(CPP11_THREADS) $(TBB_LIBS) 

clean:
	rm dd_threads
