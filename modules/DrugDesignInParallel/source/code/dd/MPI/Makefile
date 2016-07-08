# Compiles all %.cpp files into executables using mpiCC
CXX	= mpiCC -std=c++11

JOBDIR	:= jobout
SOURCES	:= $(wildcard *.cpp)
EXECS	:= $(patsubst %.cpp,%,$(SOURCES))

.PHONY: all
.PHONY: clean

all: $(EXECS)

%: %.cpp
	$(CXX) -o $@ $<

clean:
	@ rm -f $(EXECS) $(JOBDIR)/*
