
########################################

CC          = gcc
NVCC        = nvcc
NVCCFLAGS   = -O2  -DNO_DEBUG
CFLAGS      = -Wall -O2 -DNO_DEBUG
OPENMP_FLAG = -fopenmp

########################################

LIB         = -lGLEW -lglut -lGLU -lGL
CUDA_OBJ    = cuda.o cuda-kernel.o
OPENMP_OBJ  = openmp.o
COMMON_OBJ  = life.o callbacks.o fileio.o
EXE         = cuda-life openmp-life

########################################

all:openmp-life cuda-life

cuda-life:$(CUDA_OBJ) $(COMMON_OBJ)
	$(NVCC) -o $@ $(LIB) $+

openmp-life:$(OPENMP_OBJ) $(COMMON_OBJ)
	$(CC) -o $@ $(OPENMP_FLAG) $(LIB) $+

cuda.o:cuda.c
	$(NVCC) $(NVCCFLAGS) -c -o $@ $+

cuda-kernel.o:cuda-kernel.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $+

$(COMMON_OBJ): %.o: %.c
	$(CC) $(CFLAGS)  -c -o $@ $+

$(OPENMP_OBJ): %.o: %.c
	$(CC) $(CFLAGS) $(OPENMP_FLAG)  -c -o $@ $+

clean:
	rm -f $(EXE) $(CUDA_OBJ) $(OPENMP_OBJ) $(COMMON_OBJ)

.PHONY: all clean

########################################
