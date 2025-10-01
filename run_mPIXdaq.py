#!/usr/bin/env python

import os, platform, sys

# on some Linux systems, pypixet requires '.' in LD_LIBRARY_PATH to find C-libraries
#  - add current directory to LD-LIBRARY_PATH
#  - and restart python script for changes to take effect

path_modified = False
if 'LD_LIBRARY_PATH' not in os.environ and platform.system() != 'Windows':
    os.environ['LD_LIBRARY_PATH'] = '.'
    path_modified = True
    print(" ! temporarily added '.' to LD_LIBRARY_PATH !")
    # restart script in modified environment
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

# start daq in working directory
rD = mpixdaq.runDAQ(wd)
rD()
