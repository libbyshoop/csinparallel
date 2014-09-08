public class SerialPrimes {

    public static boolean isPrime(long num) {
	//returns whether num is prime

	int limit = (int) Math.sqrt(num);
	for(long i=2; i<=limit; i++) {
	    if(num % i == 0)
		return false;
	}
	return true;
    }

    public static void main(String args[]) {
	int pCount = 1;     //number of primes found (starting with 2)
	long nextCand = 3;  //next number to consider

	while(nextCand < 2000000) {
	    if(isPrime(nextCand)) {
		pCount++;
	    }
	    nextCand += 2;
	}
	
	System.out.println(pCount + " primes found");
    }
}
