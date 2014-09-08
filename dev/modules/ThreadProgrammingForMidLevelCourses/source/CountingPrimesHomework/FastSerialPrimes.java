/*
 * FastSerialPrimes.java
 *
 * Faster version of SerialPrimes.java that exploits fact that you can
 * just test for divisibility by smaller primes.
 */

import java.util.*;

public class FastSerialPrimes {

    public static ArrayList<Long> primes;

    public static boolean isPrime(long num) {
	//returns whether num is prime
	//requires that primes contains smaller primes in order
	//assumes that num is odd

	long limit = (long) Math.sqrt(num);

	long val;
	int i = 1;  //start at 1 since 2 is first and num is odd
	while((i<primes.size()) &&
	      ((val = primes.get(i)) <= limit)) {
	    if(num % val == 0)
		return false;
	    i++;
	}
	return true;
    }

    public static void main(String args[]) {
	primes = new ArrayList<Long>();
	primes.add(2L);
	long nextCand = 3;  //next number to consider
	
	while(nextCand < 2000000) {
	    if(isPrime(nextCand)) {
		primes.add(nextCand);
	    }
	    nextCand += 2;
	}
	
	System.out.println(primes.size() + " primes found");
    }
}
