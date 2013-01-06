PROG=masterWorker
CC=/usr/bin/mpicc
CFLAGS= -Wall

$(PROG): $(PROG).c
	$(CC) $(CFLAGS) $(PROG).c -o $(PROG)

clean:
	rm -f $(PROG) a.out *~ *# *.o

