from multiprocessing import *
import time

def addTwoNumbers(a, b, q):
    # time.sleep(5) # In case you want to slow things down to see what is happening.
    q.put(a+b)

def addTwoPar():
    x = input("Enter first number: ")
    y = input("Enter second number: ")

    q = Queue()
    p1 = Process(target=addTwoNumbers, args=(x, y, q))
    p1.start()