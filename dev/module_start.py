#!/usr/bin/python2
### Much of this is borrowed heavily from the quickstart.py file of the sphinx
### distribution, which is licensed under the BSD 2-clause license by the Sphinx
### team, Copyright (c) 2007-2013
import sphinx.quickstart as qs
import json
from sphinx.util.console import nocolor, color_terminal
from sys import exit

# The configuration string. Feel free to edit this.

json_conf = '''{
"makefile": true,
"ext_ifconfig": false,
"suffix": ".rst",
"ext_autodoc": false,
"author": "CSInParallel Project",
"ext_mathjax": false,
"ext_pngmath": true,
"sep": true,
"ext_todo": false,
"ext_coverage": false,
"ext_viewcode": false,
"batchfile": true,
"master": "index",
"epub": false,
"ext_intersphinx": false,
"dot": "_",
"ext_doctest": false
}'''

# Load the options from the json string into a Python dict
options = json.loads(json_conf)

# Everything below is copied from quickstart.py: you need not understand it
# unless noted

if not color_terminal():
    nocolor()

try:
    #Ask the user for the rest of the options
    qs.ask_user(options)
except (KeyboardInterrupt, EOFError):
    print
    print '[Interrupted.]'
    exit()
qs.generate(options)
