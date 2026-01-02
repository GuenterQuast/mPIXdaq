"""ADVACAM pixet1.8.4 libraries for windows 64bit

Note: Windows version needs local directory in library path

"""

import os
import sys

PIXETDIR = os.path.dirname(os.path.realpath(__file__))
print("advacam_win64: ", PIXETDIR)
# add current file directory so that pypixet.so is found by python
sys.path.append(PIXETDIR)
# change directory so that pypixet.so finds its dependencies
os.chdir(PIXETDIR)
