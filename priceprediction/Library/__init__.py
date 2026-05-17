
'''
CommonPy: an assortment of Python Library functions and utility classes

Authors
-------

Darshan Lad-darshan.l@intellectbizware.com-  Library

Copyright '''

# Package metadata ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  ╭────────────────────── Notice ── Notice ── Notice ─────────────────────╮
#  |    The following values are automatically updated at every release    |
#  |    by the Makefile. Manual changes to these values will be lost.      |
#  ╰────────────────────── Notice ── Notice ── Notice ─────────────────────╯

__version__     = '1.1.0'
__author__      = 'Darshan Lad'
__Folder__      = 'Library'

# Miscellaneous utilities.
# .............................................................................

def print_version():
    print(f'{__name__} version {__version__}')
    print(f'Authors: {__author__}')
    print(f'Folder: {__Folder__}')

