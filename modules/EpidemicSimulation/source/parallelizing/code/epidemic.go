/* Go code for CSinParallel's Epidemic Simulation module
Last updated 6-26-13 */

//This is how all Go programs begin.
package main

import ( "fmt" //short for format; needed to print to screen
		"math/rand" //for generating random numbers
		"time" //for seeding the random function
		"math" )//for square root function
		
//In Go, the type is optional and comes after the name.
const width int = 10000
const height int = 10000

//This is the Go version of an enum: the first variable's value
//is zero, and the rest follow successively.
const (
  Susceptible = iota
  Infected
  Recovered
)

//Go uses structs exclusively - there are no classes.
type Infection struct {
  duration int
  radius, contagiousness float64 //multiple variables of the same type can simply be collapsed like this
}

//another struct!
type Person struct {
	x, y, state, infectedPeriod int
}

//Methods for a struct have to be declared outside of it.
//but we can associate functions with structs by giving the struct before the function name
//and we do it as a pointer here so that the instance is passed by reference, not copied.
	func (p *Person) init() { //if the return type is void, simply don't state a type
		p.x = rand.Intn(width) //gets a random integer between zero and the parameter's value (in this case width)
		p.y = rand.Intn(height)
		p.state = Susceptible
		p.infectedPeriod = 0
	}
  
	func (p *Person) isInfected() bool { //if there's a return value, its type comes last
		return p.state==Infected //like C++, == for checking equality
	}
	
	func (p *Person) isSusceptible() bool {
		return p.state==Susceptible
	}
  
	func (p *Person) updateState(s int) {
		p.state = s
	}
	
	func (p *Person) infectWith(i Infection) {
		p.updateState(Infected)
		p.infectedPeriod = i.duration
	}
	
	func (p *Person) move() {
		p.x = (p.x + rand.Intn(5) -2 +width) % width 
		p.y = (p.y + rand.Intn(5) -2 +height) % height
	}
	
	func (p *Person) timeStep() {
		p.move()
		if p.infectedPeriod > 0 {
			p.infectedPeriod--
		//because Go uses newlines to decide where to put semicolons, the bracket style below (else on the same line as the previous bracket) is necessary for if-else statements
		} else if p.infectedPeriod == 0 && p.state == Infected {
			p.updateState(Recovered)
		}
	}

//Implicit typecasting isn't done, even between ints and floats,
// so in order to take the square root of an integer we have to cast it to a float first.
//However, we can give the type of all of our parameters at once, as shown below.
func distance(m, n, a, b int) float64 {
	ret := math.Sqrt(float64((a-m)*(a-m)+(b-n)*(b-n)))
	return ret
}

func main() {

	//parameters for the simulation: the number of people, how many should initially be infected, and how many six-hour periods to run for
	const numPersons = 20000
	const initialInfected = 200
	const numIterations = 200
  
	//seeding random with the current time (converted to Unix time, so it returns a floating-point number of seconds passed since 1/1/70). Otherwise we'd get the same outputs each time we run the simulation!
	rand.Seed(time.Now().Unix()) 

	//Go primarily uses slices rather than arrays, which are created like this
	Population := make([]Person, numPersons)

	//for loops look like this! := says to create and then initialize the variable,
	//and no parentheses are needed around the loop controls
	//However, the body must be between brackets, even when it's only one line long
	//And finally, i++, not ++i
	for i:=0; i<numPersons; i++ { Population[i].init() }

	//rather than writing a constructor, we can simply pass in values for the variables
	//and they are assigned in the order they appear in the struct's definition
	influenza := Infection{28,45,.5}

	for i := 0; i<initialInfected; i++ {
		Population[i].infectWith(influenza)
	}
	
	//This is how to print to standard out: call fmt.Println, and the elements to be printed appear in parentheses, separated by commas.
	//Spaces are automatically added between elements, and the same formatting characters apply in Go as in C++/Python
	fmt.Println("\nStarting with" , numPersons , "people, of whom are\n\tSusceptible:" , numPersons-initialInfected , "\n\tInfected:" , initialInfected , "\n\tRecovered: 0\n")

	//Here is the body of the simulation.
	//Outermost loop: run the whole thing numIterations times.
	for h := 0; h<numIterations; h++ {

		//For each Person in the array
		for i := 0; i<numPersons; i++ {
			//move, and if they're sick, decrease how much longer they'll be sick.
			Population[i].timeStep()
			//And if they are sick, for every other person in the array
			if (Population[i].isInfected()) {
				for j:=0; j<numPersons; j++ {
					//if the people are sufficiently near each other and the second Person is susceptible
					if (Population[j].isSusceptible() && distance(Population[i].x,Population[i].y,Population[j].x,Population[j].y) < influenza.radius ) {
						//use the contagiousness factor to determine whether transmission occurs
						if (rand.Intn(100) <= int(influenza.contagiousness*100)) {
							Population[j].infectWith(influenza)
						}
					}
				}
			}
		}
	}
	
	//After simulation is complete, count up the number of people in each category and print them out.
	//We'll use an array to keep track of them, since the states correspond nicely to array indices
	var numByState [3]int
	for i:= 0; i<numPersons; i++ {
		numByState[Population[i].state]++
	}
	fmt.Println("Finished! After" , numIterations/4, "days...\nNumber of persons\n\tSusceptible:" , numByState[Susceptible] , "\n\tInfected:" , numByState[Infected] , "\n\tRecovered:" , numByState[Recovered])
	  
}