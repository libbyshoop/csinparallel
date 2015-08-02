from multiprocessing import *

def sayHi():
    print "Hi from process", current_process().pid

def procEx():
    print "Hi from process", current_process().pid, "(parent process)"

    otherProc = Process(target=sayHi, args=())

    otherProc.start()

### execute
procEx()