# Release Notes for mPIXdaq

## [1.0.0] - 2026-01-06

## initial production release


## towards [1.0.1]

  - added example of bad-pixel list  
     (sn2897_badpixels.txt) for device with serial number 2897
  - added analysis prescaling
  - miniPIX readout in callback speeds up data acquisition
  - added Advacam .clog as possible input format  
  - reading from files now sequential for .yml, .txt and .clog
  - added decoder for mPIX cluster files
  - added Paused state
  - added simple GUI and option to remove keyboard control
  - run time (-t option) now refers to live time (not wall-clock time)
  - added an educator's guide
  - added library physics_tool to calculate energy deposits in matter
  - some additions (dE/dx studies) in Jupyter notebook
  - added module mpixhelpers and moved fileDecoders, bhist and scatterplot
 
## towards [1.1.0]  1.1.0rc0 and 1.1.0rc1

  - based on API 1.8.5 (released 2026/03/09)
  - supports shared memory (needs Python vers. >= 3.8)
  - fixed single pixel statistics in graphical display
  - added data samples for analysis
  - store time stamp for empty frames
  - added status field to graphics display 
  - refinements to analysis notebook
  - added section on adanced experiments to EducatorsGuide

## [1.1.0] - 2026-05-17
  release for Advacam SDK v1.8.5 and Python 3.12 for Windows

## towards [1.1.1]
  - fixed width and height of cluster bounding box
  - added grun_mpixDAQ.py and argparse_tk_gui.py for start via gui
  - improved dcoumentation
  