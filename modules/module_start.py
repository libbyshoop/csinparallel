#!/usr/bin/python2
### Much of this is borrowed heavily from the quickstart.py file of the sphinx distribution
###, which is licensed under BSD 2-clause license by the Sphinx team, Copyright (c) 2007-2013
import sphinx.quickstart as qs
import json
from sphinx.util.console import nocolor, color_terminal
from sys import exit

if not color_terminal():
    nocolor()

options = json.load(open("../../csinparallel_conf.json"))

try:
    qs.ask_user(options)
except (KeyboardInterrupt, EOFError):
    print
    print '[Interrupted.]'
    exit()
qs.generate(options)
