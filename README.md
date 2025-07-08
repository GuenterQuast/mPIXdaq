# miniPIXdaq: Data acquisition for *miniPIX EDU* pixel detector   

The [miniPIX EDU](https://advacam.com/camera/minipix-edu) is a camera
for radiation based on the [Timepix](https://home.cern/tags/timepix) 
pixel read-put chip with 256x256 radiation-sensitive pixels of 5x55µm² 
area and 300µm depth each. The chip is covered by a very thin foil to 
permit α and β radiation to reach  the pixels. The device is enclosed 
in a sturdy aluminum housing with a USB 2.0 interface.

Other than single semi-conductor chips or simple Geiger counters, 
this device provides two-dimensional images of particle traces in 
the sensitive detector material. The high spatial resolution compared 
to the typical range of particles in silicon is useful to distinguish 
the different types of radiation and measure their deposited energies. 
α-particles are completely absorbed and deposit all of their energy in 
the sensitive area, allowing usage of the device as an energy spectrometer.  

The vendor provides a ready-to-use program for different computer platforms 
as well as a software-development kit for own applications. 

The code provided here is a minimalist example to read out single
frames, i.e. a set of 256x256 measurements accumulated over a given, 
fixed time interval. Each frame is displayed as an image representation
with a logarithmic color scale representing the deposited energy. 
The analysis of the recorded signals, i.e. clustering of pixels, energy
determination and visualization, is achieved with standard open-source
tools for data analysis. It is therefore well-suited to give high-school
or university students detailed insights. 


## Getting ready for data taking

 - set up the USB interface of your computer to recognize the miniPIX EDU:  
   ``sudo install_driver_rules.sh`` (to be done only once),
   then connect the *miniPIX EDU* device to your computer  

 - include the directory with all the relevant libraries 
 (`pypixet.so`, `minipix.so` and `pxcore.so`) in the LD_LIBRARY_PATH:  
   ``export LD_LIBRARY_PATH="."``

- run the *Python* program:  
   ``./miniPIXdaq.py`` 


## Running the example script

Available options of the *Python* example to steer data taking 
and data archival to disk are shown by typing 

  ``./miniPIXdaq.py --help``; the output is shown below:

```
  usage: miniPIXdaq.py [-h] [-v VERBOSITY] [-b NUMBER_OF_BUFFERS] 
    [-a ACQ_TIME] [-c ACQ_COUNT] [-f FILE] [-t TIME]
    [--circularity_cut CIRCULARITY_CUT]  [-r READFILE]

  read, analyze and display data from miniPIX EDU device

  options:
    -h, --help            show this help message and exit
    -v VERBOSITY, --verbosity VERBOSITY
                        verbosity level (1)
    -b NUMBER_OF_BUFFERS, --number_of_buffers NUMBER_OF_BUFFERS
                        number of buffers
    -a ACQ_TIME, --acq_time ACQ_TIME
                        acquisition time/frame (0.2)
    -c ACQ_COUNT, --acq_count ACQ_COUNT
                        number of frames to add(5)
    -f FILE, --file FILE  file to store frame data
    -t TIME, --time TIME  run time in seconds
    --circularity_cut CIRCULARITY_CUT
                        cicrularity cut
    -r READFILE, --readfile READFILE
                        file to read frame data
```
The default values are adjusted to situations with low rates, where a
number of `acq_count=5` samplings with an integration time of
`acq_time = 0.2`s each are read from the device and added up to form
one frame. These frames are displayed as a two-dimensional pixel map 
with a color code indicating the energy measured in each pixel. 
For the graphics display, `number_of_buffers=2` recent frames are overlaid.
Data analysis consists of clustering of pixels in each frame and
determination of cluster parameters, like the number of pixels, energy
and circularity. The threshold on circularity is controlled by the
parameter `circularity_cut` ranging from 0. for perfectly linear 
to 1. for perfectly circular clusters. Technically, the covariance
matrix of the clusters is calculated, and the circularity is the 
ratio of the smaller and the larger of the two eigenvalues of the
covariance matrix. This simple procedure already  provides a good
separation of α and β particles and of isolated pixels not assigned
to clusters, which have a high probability of being produced in
interactions of photons.  

To test the software without access to a miniPIX EDU device or to a
radioactive source, a file with recorded data is provided. Use the
option `--readfile BlackForestStone.npy.gz` to start a demonstration.
Note that the analysis of the recorded pixel frames is done in real
time and may take some time on slow computers. 


## Implementation Details

The data acquisition is based on the function 
*doSimpleIntegralAcquisition()* from  the *ADVACAM* *Python* API,
since more advanced modes are not available for the Medipix2 chip
of the miniPIX EDU. A fixed number of frames (*qcq_counts*) with an 
adjustable accumulation time (*acq_time*) are read from the miniPIX 
device and added up. 

The chosen readout mode is *PX_TPXMODE_TOT*, where "ToT" means 
"time over threshold". This quantity is used as an approximate 
measure of the deposited energy.

Note that this is highly non-linear for small signals, but becomes
more linear at high values reaching up to 1022. Since the clock
of the miniPIC EDU chip runs at approx. 10 MHz, this corresponds 
to about 100µs maximum signal length.

Calibration constants are stored on the chip for each pixel, which
are used to provide deposited energies per Pixel in units of keV. 


## Data Analysis

The analysis shown in this example is intentionally very simple and
based on standard libraries and functions. Pixels are clustered with
the *scipi.cluster.DBSCAN* (Density-Based Spatial Clustering of Applications
with Noise). The shape of the clusters is determined from the ratio of
the smaller and the larger one of the two eigenvalues of the covariance 
matrix, which is calculated from the *x* and *y* coordinates of the pixels
in a cluster. For circular clusters, as produced by α radiation, this ratio
is close to one, while it is almost zero for the longer traces from β radiation. 

The figure below shows the graphical display of the program and the
typical distribution of the pixel and cluster energies and the
number of pixels per cluster. The radioactive source used was a weakly
radioactive stone from the Black Forest containing a small amount of
Uranium and its decay products. The pixel map shown in the figure was 
sampled over a time of two seconds. The histogram in the lower-right
corner shows how cluster types distinguish different types of radiation. 

![The graphical display of miniPIXdaq](miniPIXdaq.png)


## Sensor Details

The miniPIX EDU is based on the *Timepix" hybrid silicon pixel device, 
consisting of a semiconductor detector chip segmented into 
256 x 256 square pixels with a pitch of 55 mm that is bump-bonded to 
the readout chip. Each element of the pixel matrix is connected to its
own preamplifier, discriminator and digital counter integrated on the
readout chip. 

The built-in Medipix2 variant of the chip is operated in the so-called
"frame mode", i.e. all pixels are read out at the same time, providing
one frame consisting of the deposited energies per pixel collected during
the acquisition time. If operated in time-over-threshold (ToT) mode,
returned pixel readings represent the time the signal is over 
a given threshold in counts of the chip clock (appr. 10 MHz). 
*ToT* is linearly related to the energy deposition for large deposits 
exceeding 50 keV. The functional dependence on the deposited energy $E$, 
including threshold effects, is approximated by the following function

   $ToT\,=\;a\,E +b - {c}/{(E-t)}$

Approximate values of the calibrations constants are 
$a$ = 1.6, $b$=23, $c$=23 and $t$=4.3. 
Each pixel has its individual calibration stored on the chip, 
which is optionally applied. The calibration is reliable up to 
pixel energies of one MeV. Higher pixel  energies may result when 
frames with short acquisition time are summed up. For details, 
see the article by *J. Jakubek, "Precise energy calibration of 
pixel detector working in time-over-threshold mode", 
NIM A 633 (2011), 5262-5265*.
