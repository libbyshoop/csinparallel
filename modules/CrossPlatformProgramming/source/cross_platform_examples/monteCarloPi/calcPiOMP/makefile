OBJ = calcPiOMP

CC = g++
CFLAGS = -Wall -ansi -pedantic -std=c++11
LFLAGS = -o $(OBJ) -fopenmp

$(OBJ): $(OBJ).C
	$(CC) $(CFLAGS) $(OBJ).C $(LFLAGS)

clean:
	rm -f a.out $(OBJ)
