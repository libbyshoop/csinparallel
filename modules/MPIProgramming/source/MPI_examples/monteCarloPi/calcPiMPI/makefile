PROG = calcPiMPI
CC = mpicxx
CFLAGS = -Wall -ansi -pedantic -std=c++11
LFLAGS = -o $(PROG)

$(PROG): $(PROG).C
	$(CC) $(CFLAGS) $(PROG).C $(LFLAGS)

clean:
	rm -f $(PROG) a.out *~ *# *.o

