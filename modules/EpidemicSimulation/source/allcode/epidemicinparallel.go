/* Go code for CSinParallel, last updated 6-26-13 */

//This is how all Go programs begin.
package main


//Note: these could all be combined into a single statement, unlike in C++
import ( "fmt" //short for format; needed to print to screen
		"math/rand" //for generating random numbers
		"time" //for seeding the random function
		"math" //for square root function
		"sync" ) //for parallelizing

//In Go, the type is optional and comes after the name.
//these variables represent the dimensions of the world, the number of people in it,
//how many of them are initially infected, and how many six-hour periods to iterate over
const width int = 10000
const height int = 10000
const numPersons = 20000
const initialInfected = 200
const numIterations = 200

var waitgr sync.WaitGroup
  
//This is the Go version of an enum: the first variable's value
//is zero, and the rest follow successively.
const (
  Susceptible = iota
  Infected
  Recovered
  Dead
)

//Go uses structs exclusively - there are no classes.
type Infection struct {
  duration int
  radius, contagiousness, deadliness float64 //multiple variables of the same type can simply be collapsed like this
}

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
	
	func (p *Person) infectWith(inf Infection) {
		p.updateState(Infected)
		p.infectedPeriod = inf.duration
	}
	
	func (p *Person) move() {
		p.x = (p.x + rand.Intn(5) -2 +width) % width 
		p.y = (p.y + rand.Intn(5) -2 +height) % height
	}
	
	func (p *Person) timeStep(inf Infection) {
		p.move()
		if p.isInfected() {
			if p.infectedPeriod == 0 {
				p.updateState(Recovered)
			} else if (rand.Intn(100) <= int(inf.deadliness*100)) {
				p.updateState(Dead)
			} else {
				p.infectedPeriod--
			}
		}
	}

//Implicit typecasting isn't done, even between ints and floats,
// so in order to take the square root of an integer we have to cast it to a float first.
func distance(m, n, a, b int) float64 {
	ret := math.Sqrt(float64((a-m)*(a-m)+(b-n)*(b-n)))
	return ret
}

func collectInfected(i int, Population *[]Person, infectedchan chan Person) {
	if (*Population)[i].isInfected() {
			infectedchan <- (*Population)[i]
	}
	waitgr.Done()
}

func iterateThruInfected(infectedchan chan Person, Population *[]Person, influenza Infection) {
	for p := range infectedchan {
		go infectNeighbors(Population, influenza, p)
	}
}
	
func infectNeighbors(Population *[]Person, influenza Infection, p Person) {
	for i:=0; i<numPersons; i++ {
		if ((*Population)[i].isSusceptible() && distance(p.x,p.y,(*Population)[i].x,(*Population)[i].y) < influenza.radius) {
			if (rand.Intn(100) <= int(influenza.contagiousness*100)) {
					(*Population)[i].infectWith(influenza)
			}
		}
	}
}

func main() {

	rand.Seed(time.Now().Unix()) 

	Population := make([]Person, numPersons)
	for i:=0; i<numPersons; i++ { Population[i].init() }
	
	influenza := Infection{28,45,.5,.3}
	for i := 0; i<initialInfected; i++ {
		Population[i].infectWith(influenza)
	}

	infectedchan := make(chan Person, numPersons)

	fmt.Println("\nStarting with" , numPersons , "people, of whom are\n\tSusceptible:" , numPersons-initialInfected , "\n\tInfected:" , initialInfected , "\n\tRecovered: 0\n\tDead: 0\n")

//--------------begin simulation--------------------
	for h := 0; h<numIterations; h++ {

		for i:= 0; i<numPersons; i++{
			Population[i].timeStep(influenza)
			waitgr.Add(1)
			go collectInfected(i, &Population, infectedchan)
		}

		waitgr.Wait()
		iterateThruInfected(infectedchan, &Population, influenza)

	}

	var numByState [4]int
	for i:= 0; i<numPersons; i++ {
		numByState[Population[i].state]++
	}
	fmt.Println("Finished! After" , numIterations/4, "days...\nNumber of persons\n\tSusceptible:" , numByState[Susceptible] , "\n\tInfected:" , numByState[Infected] , "\n\tRecovered:" , numByState[Recovered], "\n\tDead:", numByState[Dead])
}