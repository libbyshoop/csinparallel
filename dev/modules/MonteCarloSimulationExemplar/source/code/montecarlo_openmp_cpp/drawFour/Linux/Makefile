seq:
	g++ -o drawFourSuits_seq drawFourSuits_seq.cpp
	
omp:
	g++ -o drawFourSuits_omp drawFourSuits_omp.cpp -fopenmp
	
omp_seeds:
	g++ -o drawFourSuits_omp_seeds drawFourSuits_omp_seeds.cpp -fopenmp

all: seq omp omp_seeds

clean:
	rm drawFourSuits_seq drawFourSuits_omp drawFourSuits_omp_seeds