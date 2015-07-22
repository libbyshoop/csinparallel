from multiprocessing import *

def sayHi4(lock, name):
    lock.acquire()
    print "Hi", name, "from process", current_process().pid
    lock.release()

def manyGreetings4():
    lock1 = Lock()
    
    print "Hi from process", current_process().pid, "(main process)"
    
    for i in range(10):
        Process(target=sayHi4, args=(lock1, "p"+str(i))).start()

#execute
manyGreetings4()