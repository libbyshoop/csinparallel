PROG=ring
#CC=/Developer/usr/bin/i686-apple-darwin9-gcc-4.2.1
#CC=/Developer/usr/bin/mpicc
CC=/usr/bin/mpicc
CFLAGS= -Wall

$(PROG): $(PROG).c
	$(CC) $(CFLAGS) $(PROG).c -o $(PROG)

clean:
	rm -f $(PROG) a.out *~ *# *.o

