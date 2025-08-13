#!/usr/bin/env python3
import os, sys

# pypixet requires '.' in LD_LIBRARY_PATH so that al neccessary C-libraries are found
#  - add current directory to LD-LIBRARY_PATH
#  - and restart python script for changes to take effect

path_modified = False

if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = '.'
    path_modified = True
elif not '.' in os.environ['LD_LIBRARY_PATH']:
    os.environ['LD_LIBRARY_PATH'] += ':.'
    path_modified = True

if path_modified:
    print(" ! temporarily added '.' to LD_LIBRARY_PATH !")
    try:
        os.execv(sys.argv[0], sys.argv)
    except Exception as e:
        sys.exit('EXCEPTION: Failed to Execute under modified environment, ' + str(e))
else:  # restart python script for setting to take effect
    # get current working directory before importing minipix libraries
    wd = os.getcwd()
    from mpixdaq import mpixdaq  # this changes the working directory!

    rD = mpixdaq.runDAQ(wd)  # start daq in working directory
    rD()
