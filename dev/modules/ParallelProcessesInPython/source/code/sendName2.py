from multiprocessing import *
import time
from random import randint


def greet2(q):
    for i in range(5):
        print
        print "(child process) Waiting for name", i
        name = q.get()
        print "(child process) Well, hi", name

def sendName2():
    q = Queue()
   
    p1 = Process(target=greet2, args=(q,))
    p1.start()

    for i in range(5):
        time.sleep(randint(1,4))
        print "(main process) Ok, I'll send the name"
        q.put("George"+str(i))

#execute
sendName2()
