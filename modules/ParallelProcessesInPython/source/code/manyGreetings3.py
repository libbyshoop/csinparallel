from multiprocessing import *

def sayHi3(personName):
    print "Hi", personName, "from process", current_process().name, "- pid", current_process().pid

def manyGreetings3():
    print "Hi from process", current_process().pid, "(parent process)"
    
    personName = "Jimmy"
    for i in range(10):
        Process(target=sayHi3, args=(personName,), name=str(i)).start()

#execute
manyGreetings3()
