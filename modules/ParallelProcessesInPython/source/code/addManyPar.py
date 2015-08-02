from multiprocessing import *
from random import randint
import time
def addManyNumbers(numNumbers, q):
    s = 0
    for i in range(numNumbers):
        s = s + randint(1, 100)
    q.put(s)

def addManyPar():
    totalNumNumbers = 1000000

    q = Queue()
    p1 = Process(target=addManyNumbers, args=(totalNumNumbers/2, q))
    p2 = Process(target=addManyNumbers, args=(totalNumNumbers/2, q))
    p1.start()
    p2.start()

    answerA = q.get()
    answerB = q.get()
    print "Sum:", answerA + answerB

#execute
addManyPar()
