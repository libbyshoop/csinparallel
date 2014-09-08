/*
 * ThreadedPrimes.java
 *
 * Version of threaded prime counter to give students.  Doesn't (yet) work.
 */

public class ThreadedPrimes {

    public static int pCount;
    public static Object lock;

    static class PrimeFinder implements Runnable {

	private long from;
	private long to;
	public long pCountIndv;

	public PrimeFinder(long From, long To) {
	    from = From;
	    to = To;
	}

	public void run() {
	    long nextCand = from;  //next number to consider

	    while(nextCand < to) {
	        if(isPrime(nextCand)) {
		    //pCountIndv++;
		    pCount++;
		}
		nextCand += 2;
	    }
	    //pCount += pCountIndv;
	}
    }

    public static boolean isPrime(long num) {
	//returns whether num is prime

	int limit = (int) Math.sqrt(num);
	for(long i=2; i<=limit; i++) {
	    if(num % i == 0)
		return false;
	}
	return true;
    }

    public static void main(String args[]) throws InterruptedException {
	pCount = 1;   //(starting with 2)

	PrimeFinder p1 = new PrimeFinder(3, 1300000);
	Thread t1 = new Thread(p1);
	PrimeFinder p2 = new PrimeFinder(1300001, 2000000);
	Thread t2 = new Thread(p2);

	t1.start();
	t2.start();
	t1.join();
	t2.join();

	System.out.println(pCount + " primes found");
    }
}
