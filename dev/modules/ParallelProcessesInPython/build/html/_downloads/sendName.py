from multiprocessing import *
import time

def greet(q):
    print "(child process) Waiting for name..."
    name = q.get()
    print "(child process) Well, hi", name

def sendName():
    q = Queue()
    
    p1 = Process(target=greet, args=(q,))
    p1.start()
    
    time.sleep(5) # wait for 5 seconds
    print "(parent process) Ok, I'll send the name"
    q.put("Jimmy")

#execute
sendName()
