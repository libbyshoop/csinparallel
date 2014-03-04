OBJ    = critical3

CC     = g++
CFLAGS = -Wall -ansi -pedantic 
LFLAGS = -o $(OBJ) -fopenmp

$(OBJ): $(OBJ).cpp
	$(CC) $(CFLAGS) $(OBJ).cpp $(LFLAGS)

clean:
	rm -f $(OBJ)

