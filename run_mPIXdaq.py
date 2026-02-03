#!/usr/bin/env python
#
# script run_mPIXdaq.py
#  run mpixdaq example with data acquisition, on-line analysis and visualization
#  of pixel frames and histogramming

import os
import platform
import sys

# on some Linux systems, pypixet requires '.' in LD_LIBRARY_PATH to find C-libraries
#  - add current directory to LD-LIBRARY_PATH
#  - and restart python script for changes to take effect

modified_path = False
if platform.system() != 'Windows':
    _ldp = os.environ.get("LD_LIBRARY_PATH")
    if _ldp:
        if ':.' not in _ldp and _ldp != '.':
            os.environ["LD_LIBRARY_PATH"] = _ldp + ':.'
            modified_path = True
    else:
        os.environ['LD_LIBRARY_PATH'] = '.'
        modified_path = True

    # restart script in modified environment
    if modified_path:     
        print(" ! temporarily added '.' to LD_LIBRARY_PATH !")
        try:
            os.execv(sys.argv[0], sys.argv)
        except Exception as e:
            sys.exit('!!! run_mPIXdaq: Failed to Execute under modified environment: ' + str(e))

# get current working directory (before importing minipix libraries)
wd = os.getcwd()

if os.name == 'nt':
    # special hack for windows python 3.7: load pypixet and DLLs
    import mpixdaq.advacam_win64.pypixet as pypixet
from mpixdaq import mpixdaq  # this may change the working directory, depending on system

# finally, start daq in working directory
rD = mpixdaq.runDAQ(wd)
rD()
