# Eileen King, June 2013, for CSinParallel

from __future__ import print_function
import sys, fileinput

modname = raw_input("Please enter your module's name: ")
projdes = raw_input("Describe your module in one sentence: ")

change = dict(extensions=r"['sphinx.ext.pngmath']",
              copyright=r"u'This work is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License'",
              html_title=r"'"+modname+r"'",
              html_logo=r"'../../../images/CSInParallel200wide.png'",
              html_favicon=r"'../../../images/favicon.ico'",
              html_show_sourcelink=False,
    latex_documents = r"""[
('index', '""" + modname + r".tex', u'" + modname + r"""',
u'CSInParallel Project', 'manual'),
]""",
    man_pages = "[\n('index', "+ "'" + modname + "', u'" + modname + "',\n[u'CSInParallel Project'], 1)\n]",
    texinfo_documents = r"""[
('index', '""" + modname + r"', u'""" + modname + r"""',
u'CSInParallel Project', '""" + modname + r"', '" + projdes + r"""',
'Miscellaneous'),
]""")

commentout = dict(version=r'1',release=r'1')

latex_el = r"""'releasename': '', 'classoptions': ',openany,oneside', 'babel' : '\\usepackage[english]{babel}'
}"""

for line in fileinput.input(inplace=1):
    entry, sep, value = line.partition(" = ")
    if entry.isspace() or 'import' in entry:
        print(line)
    elif entry.startswith("#'preamble'"):
        print(line,'\n',latex_el,'\n')
    elif entry.startswith('#'):
        newentry = entry.lstrip('#')
        if newentry in change:
            print(newentry, sep, change[newentry], sep='')
        elif newentry in commentout:
            print('#'+line)
        else:
            print(line, end='')
    else:
        if not value:
            print(end='')
        elif entry in change:
            print(entry, sep, change[entry], sep='')
        elif entry in commentout:
            print('#'+line)
        else:
            print(line, end='')

fileinput.close()
