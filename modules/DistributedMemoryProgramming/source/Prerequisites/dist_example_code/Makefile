PROGS = example2 hellompi \
	matrix_multiplication matrix_multiplication_todo \
	seq_pi_done mpi_pi_done \
	vector_matrix_buggy_done vector_matrix_buggy_todo \
	vector_matrix_mpi vector_matrix_mpi_todo

CC = mpicc
CFLAGS = -Wall
LFLAGS = -o $@


all: $(PROGS)


clean:
	rm -f $(PROGS)
