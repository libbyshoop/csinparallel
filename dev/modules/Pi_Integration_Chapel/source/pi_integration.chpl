use Time;
// config variables can be changed while running the executable, for example:
// ./pi --numRect=1000 --feedback=true --method=linear

config var numRect: int = 10000000;     // Specifies the number of rectangles to calculate
config var numThreads: int = 8;        // Specifies the number of threads to run
config var feedback: bool = false;     // Toggles text displaying partial sums or global sum for certain proceedures
config var method: string = "reduceThread";  // Chooses the algorithm to calculate pi. Options include:
       	   	   	    	       // linear, forall, forallSync, beginRace, begin, reduceRect, reduceThread 	       	  
var globalSum: real = 0.0;
var lock: sync bool;

proc areaLinear() {  

  // Calculates the area under half a curve without any parallel processing.
     
     var width: real = 2.0 / numRect;
     var x: real = -1 + width/2;
     for i in 1..numRect-1 {
     	    x = -1 + ( i + 0.5) * width;
	    globalSum += sqrt(1.0 - x*x) * width;	    
     }
}

proc areaForall() { 

  /* Calculates the area under half a curve using a forall loop. 
     Notice that it has a race condition, and has different levels
     of accuracy and precision depending on the size of numRect.
  */

     var width: real = 2.0 / numRect;
     var x: real = -1 + width/2;
     forall i in 1..numRect-1 {
   	    x = -1 + ( i + 0.5) * width;
	    globalSum += sqrt(1.0 - x*x) * width;	    
     }
}

proc areaForallSync() { 
  /* Calculates the area under half a curve using a forall loop. 
     Notice that it has a race condition, and has different levels
     of accuracy and precision depending on the size of numRect.
  */

     var sum: sync real = 0.0;
     var width: real = 2.0 / numRect;
     var x: real = -1 + width/2;
     forall i in 1..numRect-1 {
   	    x = -1 + ( i + 0.5) * width;
	    sum += sqrt(1.0 - x*x) * width;	    
     }
     globalSum = sum;
}

proc areaBegin(init) { 

  // Calculates the area under the curve for 1/numThreads worth of rectangles
  // and adds the result to globalSum.
  
     var width: real = 2.0 / numRect;
     var partialSum: real = 0.0;
     var x: real = -1 + width/2;
     var i: int = init;
     do {
   	    x = -1 + ( i + 0.5) * width;
	    partialSum += sqrt(1.0 - x*x) * width;	    
	    i += numThreads;
     }  while (i < numRect-1); 
     lock = true; // set sync variable to full
     globalSum += partialSum;
     var unlock = lock; // empty the sync variable to allow the program to move on
     if (feedback) {writeln("Thread: ", init, "   globalSum: ", globalSum);}  
}

 proc rectArea(i : int) { 

   // Calculates the area for one rectangle, so that a reduce operator
   // can calculate areas for all the rectangles in a domain.
  
     var width: real = 2.0 / numRect;
     var x: real = -1 - width/2 + i*width;
     return width * sqrt(1.0-x*x);         
}

proc areaReduceThread(init : int) { 

  // Calculates the area under the curve for 1/numThreads worth 
  // of rectangles and returns a partial sum
     
     var width: real = 2.0 / numRect;
     var partialSum: real = 0.0;
     var x: real = -1 + width/2;
     var i: int = init;
     do {
   	    x = -1 + ( i + 0.5) * width;
	    partialSum += sqrt(1.0 - x*x) * width;	    
	    i += numThreads;
     }  while (i < numRect-1);
     if (feedback) { writeln(partialSum);} 
     return partialSum;         
}


var timer: Timer;
timer.start();

if (method == "linear") {
    areaLinear();     
} else if (method == "forall") {
    areaForall();
} else if (method == "forallSync") {
    areaForallSync();
} else if (method == "beginRace") {
    for i in 1..numThreads {
      begin areaBegin(i);     
    }
} else if (method == "begin") {
    sync for i in 1..numThreads { 
      // The sync statements ensures that tasks created in the for loop 
      // must complete before continuing past the sync statement.     
      begin areaBegin(i);     
    }
} else if (method == "reduceRect") {
    var partials: domain(1) = (1..numRect);
    globalSum = + reduce (rectArea(partials));
} else if (method == "reduceThread") {
    var threads: domain(1) = (1..numThreads);
    globalSum = + reduce (areaReduceThread(threads));   
}  else {
    writeln("Invalid method name. Please choose from: linear, forall, forallSync, beginRace, begin, reduceRect, or reduceThread .");
 }

timer.stop();

var pi: real = globalSum*2.0;
writeln("This program estimates pi as ", pi);
writeln("The operation took ", timer.elapsed(), " seconds.");
