CC     = g++
CFLAGS = -Wall -ansi -pedantic -std=c99
NVCC = nvcc

devices: devices.cu
	$(NVCC) $(CFLAG) devices.cu -o devices 

all: devices

clean: 
	rm -rf *o devices 
