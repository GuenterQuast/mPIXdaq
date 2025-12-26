## mPIXdaq: Data acquisition for *miniPIX (EDU)* pixel detector 
----------------------------------------------------------------  

                                            Vers. 1.0.0b, November 2025

The [miniPIX EDU](https://advacam.com/camera/minipix-edu) is a camera
for radiation based on the [Timepix](https://home.cern/tags/timepix) 
pixel read-out chip with 256x256 radiation-sensitive pixels of 55x55µm² 
area and 300µm depth each. The chip is covered by a very thin foil to 
permit α and β radiation to reach  the pixels. The device is enclosed 
in an aluminum housing with a USB 2.0 interface. The sensor chip
is covered by a thin foil and is very fragile; this area should be 
covered with a protective material if not measuring α radiation. 

Other than single semi-conductor chips or simple Geiger counters, 
this device provides two-dimensional images of particle traces in 
the sensitive detector material. The high spatial resolution compared 
to the typical range of particles in silicon is useful to distinguish 
the different types of radiation and measure their deposited energies. 
α-particles are completely absorbed and deposit all of their energy in 
the sensitive area, allowing usage of the device as an energy spectrometer.  

The vendor provides a ready-to-use program for different computer
platforms as well as a software-development kit for own applications. 

The code provided here is a minimalist example to read out single
frames, i.e. a full set of 256x256 measurements accumulated over a 
given, fixed time interval. Each frame is displayed as an image with
a logarithmic color scale representing the deposited energy. 
The analysis of the recorded signals, i.e. clustering of pixels, energy
determination and visualization, is achieved with standard open-source
tools for data analysis. It is therefore well-suited to give high-school
or university students detailed insights and to enable them to carry out
their own studies.


## Getting ready for data taking

This code has been tested on *Ubuntu*, *openSuse*, *Fedora*, on
Windows 64bit with *Python3.7.9* and on *Raspberry Pi* for the 
32- and 64-bit versions of *OS12* and *OS13*. Other Linux distributions 
should not pose any unsurmountable problems.  
On MS Windows, the libraries provided by the vendor only support
*Python* vers. 3.7.9; such a rather historic version can be set up
using e.g. the *miniconda* framework. 

The code also supports devices other than the miniPIX EDU if the 
configuration files are available and copied to the *factory/*
directory in the *pypixet* *Python* interface. 

To get started, follow the steps below: 

 - Get the code from gitlab@KIT or from github  
   ``git clone https://gitlab.kit.edu/Guenter.Quast/mPIXdaq`` or    
   ``git clone https://github.com/GuenterQuast/mPIXdaq``.

   This repository includes the *Python* code and a minimalistic set
   of libraries provided by ADVACAM.

 - Next, `cd` to the `mPIXdaq` directory you just downloaded.

 - Set up the USB interface of your computer to recognize the miniPIX EDU:  
   ``sudo install_driver_rules.sh`` (to be done only once),
   then connect the *miniPIX EDU* device to your computer.  

The package may also be installed in your virtual python environment:

  - `python -m pip install .`


Now everything is set up to enjoy your miniPIX EDU. Just run the *Python* 
program from any working directory by typing   

   > ``run_mPIXdaq.py``.

If you plan to record data, note that the path to the output file
is relative to the current working directory. 

*Note* that the *pypixet* initialization is set up to write log-files
and configuration data to the directory */tmp/mPIX/*.

It is also worth mentioning that on some systems the current directory,
".", needs to be contained in the `LD_LIBRARY_PATH` so that the ADVACAM 
*Python* interface *pypixet* finds all its *C* libraries. This is also 
done in the *Python* script ``run_mPIXdaq.py`` by temporarily modifying 
the environment variable `LD_LIBRARY_PATH` if necessary and then restarting 
to execute the *Python* code in the new environment.  
Starting the *Python* code by a different mechanism may not work without 
adjusting the environment variable `LD_LIBRARY_PATH`. In the *bash* shell, 
this is achieved by `export LD_LIBRARY_PATH='.'` on the command line. 
Note, however, that such a permanent change opens up a security gap on 
your computer!
 

## Running the example script

Available options of the *Python* example to steer data taking and 
data archival to disk are shown by typing 

  ``run_mPIXdaq.py --help``, resulting in the following output:

```
usage: run_mPIXdaq.py [-h] [-v VERBOSITY] [-o OVERLAY] [-a ACQ_TIME] [-c ACQ_COUNT] [-f FILE] [-w WRITEFILE] [-t TIME]
                      [--circularity_cut CIRCULARITY_CUT] [-r READFILE]

read, analyze and display data from miniPIX device

options:
  -h, --help            show this help message and exit
  -v VERBOSITY, --verbosity VERBOSITY
                        verbosity level (1)
  -o OVERLAY, --overlay OVERLAY
                        number of frames to overlay in graph (10)
  -a ACQ_TIME, --acq_time ACQ_TIME
                        acquisition time/frame (0.1)
  -c ACQ_COUNT, --acq_count ACQ_COUNT
                        number of frames to add (5)
  -f FILE, --file FILE  file to store frame data
  -w WRITEFILE, --writefile WRITEFILE
                        csv file to write cluster data
  -t TIME, --time TIME  run time in seconds
  --circularity_cut CIRCULARITY_CUT
                        cut on cicrularity for alpha detection
  --flatness_cut FLATNESS_CUT
                        cut on flatness for alpha detection
  -r READFILE, --readfile READFILE
                        file to read frame data
  -b BADPIXELS, --badpixels BADPIXELS
                        file with bad pixels
```

The default values are adjusted to situations with low rates, where
frames from the *miniPIX* with an integration time of `acq_time = 0.5` s
are read. For the graphics display, `overlay = 10` recent frames are 
overlaid, leading to a total integration time of 5 s. 
These images represent a two-dimensional pixel map with a color code 
indicating the energy measured in each pixel. 

The miniPIX EDU version, in particular, may suffer from a large number of
dead or noisy pixels, and therefore they should be masked by providing a file
with the pixel indices to be ignored. The default file name is *badpixels.txt*
in the working directory; alternatively a file name may be specified using
the `-b` or `--badpixels` option. 

Collected frame data may be directly written to disk, if a filename is
given using the `-f`or `--file` option. Two formats are foreseen at present,
storage of the two-dimensional frames as numpy-arrays (file extension `.npy`) 
or as lists of pixel indices and energy values in *yaml*-format (file extension
`.yml`). To save space, the resulting output files may be compressed with
*zip' or *gzip*. If no suffix for the filename is given, the default behavior 
is writing a *.yml* file. The `.yml` format also permits storing meta data
like parameters of the data acquisition, the properties of the sensor and
the list of bad pixels.

The same formats are recognized when reading back files
using the `-r` resp. `--readfile` options. 
In addition, `.txt` files written with the *Pixet* program of Advacam
can be used as an input. 

Data analysis consists of clustering of pixels in each overlay-frame and
determination of cluster parameters, like the number of pixels, energy
of clusters, and the shapes of the cluster areas and of the energy 
distribution over the pixels in the clusters.

The shape of the cluster area is encoded in a quantity called circularity. 
For discrimination between linear and round clusters, a cut is controlled 
by the parameter `circularity_cut` ranging from 0. for perfectly linear 
to 1. for perfectly circular clusters. Technically, the covariance
matrix of the cluster area is calculated, and the circularity is defined 
as the ratio of the smaller and the larger of the two eigenvalues of the
covariance matrix. This simple procedure already provides a good
separation of α and β particles and of isolated pixels not assigned
to clusters. The latter ones have a high probability of being produced in
interactions of photons, while electrons from β radiation or from 
photon interactions produce long traces, and α particles produce large,
circular clusters due to their very high ionization loss in the
detector material.  

A further, very sensitive variable is the variance of the energy distribution
in the clusters. For α particles, this distribution peaks at the centre and
steeply falls off towards the boundary, leading to a very small variance.
A small ratio of the variances of the energy distribution and of the area
covered by pixels is therefore a very prominent signature of α particles.
The cut separating flat and peaking signatures is controlled by the parameter
`flatness` with values between 0 and 1. 

Properties of clusters are optionally written to a file in *.csv* format
for later off-line analysis. A *Jupyter* notebook, 
*analyze_mPIXclusters.ipynb*, is provided which illustrates an example 
analysis. 

To test the software without access to a miniPIX device or without
a radioactive source, a file with recorded data is provided. Use the
option `--readfile data/BlackForestStone.yml.gz` to start a demonstration.
Note that the analysis of the recorded pixel frames is done in real
time and may take some time on slow computers. 


## Implementation Details

The default data acquisition is based on the function 
*doSimpleIntegralAcquisition()* from  the *ADVACAM* *Python* API.
A fixed number of frames (*acq_counts*) with an adjustable accumulation
time (*acq_time*) are read from the miniPIX device and added up. 

The chosen readout mode is *PX_TPXMODE_TOT*, where "ToT" means 
"time over threshold". This quantity shows good proportionality to
the deposited energy at high signal values, but exhibits a strong 
non-linear behavior for small signals near the detection threshold 
of the miniPIX. Calibration constants are stored on the miniPIX
device for each pixel, which are used to provide deposited energies
per pixel in units of keV. 

The relevant libraries for device control are provided in directories
`advacam_<arch>` for `x86_64` Linux, `arm32` and `arm64` and for 
Macintosh arm64 amd MS Windows architectures. The contents of a 
typical directory is: 

```
  __init__.py   # package initialization
  pypixet.so    # the Pixet Python interface
  minipix.so    # C library for pypixet
  pxcore.so     # C library for pypixet
  pixet.ini     # initialization file, in same directory as pypixet
  factory/      # initialization constants 
```

Note that the copyright of these libraries belongs to ADVACAM. 
The libraries may be downloaded from their web page, 
[ADVACAM DWONLOADS](https://advacam.com/downloads/). 
They are provided here as *Python* packages for some platforms
for convenience. 


## Data Analysis

The analysis shown in this example is intentionally very simple and based 
on standard libraries and functions. Clustering of pixels is performed by
finding connected regions in the pixel image with *scipy.ndimage.label()*.
The shape of the clusters is determined from the ratio of the smaller 
and the larger one of the two eigenvalues of the covariance matrix 
calculated from the *x* and *y* coordinates of the pixels in a cluster 
using *numpy.cov()*. For circular clusters, as typically produced by 
α radiation, this ratio  is close to one, while it is almost zero for 
the longer traces from β radiation. In addition, she shape of the energy distribution is considered, which shows a sharp maximum at the center for
α particles but is rather flat otherwise.

The figure below shows the graphical display with a pixel image and 
the typical distributions of the pixel and cluster energies and the 
number of pixels per cluster. The source used was a weakly radioactive
stone from the Black Forest containing a small amount of Uranium and 
its decay products. The pixel map shown in the figure was sampled 
over a time of five seconds. The histogram in the lower-right
corner shows how well the cluster types discriminate different types
of radiation: α rays in the green band with relatively low numbers 
of pixels per cluster, electrons (β) as long tracks with large numbers
of pixels per cluster and rather low energies. Single pixels not 
associated to clusters mostly originate from 𝛾 rays. Some of the electron 
tracks  with typically low energies also stem from photon interactions 
in the detector material (via the Compton process).

![The graphical display of miniPIXdaq](miniPIXdaq.png)

The analysis shown here is suitable for low-rate scenarios, e.g.
the analysis of natural radiation as emitted by minerals like
Pitchblend (=Uraninit),  Columbit, Thorianit and others. Radon
accumulated from the air in basement rooms on the surface
of an electrostatically charged ballon also work fine. Therefore,
the frame collection is chosen to be on the order of seconds, 
so that analysis results can be displayed in real-time on 
a sufficiently fast computer including the Raspberry Pi 5.

For applications at higher rates, the analysis may have to
be done off-line by reading data from recorded files, or 
multiple cores must be used for the analysis task.  


## Sensor Details

The miniPIX (EDU) is based on the [*Timepix*](https://home.cern/tags/timepix)
hybrid silicon pixel device, consisting of a semiconductor detector chip
segmented into 256 x 256 square pixels with a pitch of 55 mm that is 
bump-bonded to the readout chip. Each element of the pixel matrix is
connected to its own preamplifier, discriminator and digital counter
integrated on the readout chip. 

The built-in *Medipix2* variant of the chip is operated in the so-called
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
which is optionally applied to obtain pixel readings in units of keV.
The calibration is reliable up to pixel energies of one MeV. 
Higher pixel  energies may result when frames with short acquisition 
time are summed up. For details, see the article by J. Jakubek, 
*Precise energy calibration of pixel detector working in time-over-threshold
mode*, NIM A 633 (2011), 5262-5265*.


## Package Structure

This package consists of one *Python* file with several classes providing 
the base functionality. As mentioned above, it relies on
[ADVACAM libraries](https://wiki.advacam.cz/wiki/Python_API)
for setting-up and reading the sensor. 
Other dependencies are well-known libraries from the "Python" eco-system 
for data analysis:

  - `numpy`
  - `matplotlib`,
  - `scipy.ndimage.label`
  - `numpy.cov`
  - `numpy.linalg.eig`

The classes and scripts of the package are

  - class `miniPIXdaq`
  - class `frameAnalyzer`
  - class `miniPIXvis` 
  - class `runDAQ`
  - class `bhist`
  - class `scatterplot`
  - package script `run_mPIXdaq.py`

Details on the interfaces are given below.

```
class miniPIXdaq:
    """Initialize, readout miniPIX device and store data

    After initialization, data from the device is stored in a
    ring buffer and the current buffer index is sent to the
    calling process via a Queue in an infinite loop, which
    ends when data is entered in a command Queue.

    Args:

      - ac_count: number of frames to overlay
      - ac_time: acquisition time

    Queues for communication and synchronization

      - dataQ:  Queue to transfer data
      - cmsQ: command Queue

    Data structure:

       - fBuffer: ring buffer with recent frame data

    """
```

```
class frameAnalyzer:
  """Analyze frame data
    - find clusters
    - compute cluster energies
    - compute position and covariance matrix of x- and y-coordinates
    - analyze cluster shape (using eigenvalues of covariance matrix)
    - construct a tuple with cluster properties
    - optionally write cluster data to a file in csv format 

    Note: this algorithm only works if clusters do not overlap!

    Args of __call__() method:  a 2d-frame from the miniPIX

    Returns:
     
    - self.pixel_clusters: list of tuples with properties per cluster: mean of x and y
      coordinates, the number of pixels, energy, eigenvalues of covariance matrix and
      their orientation as an angle in range [-pi/2, pi/2] and the minimal and maximal
      eigenvalues of the covariance matrix of the energy distribution. The format is:
      ( (x, y), n_pix, energy, (var_mx, var_mn), angle, (xEm, yEm), (varE_mx, varE_mn) )

    - self.cluster_pxl_lst is a list of dimension n_clusters + 1 and contains the pixel
      indices contributing to each of the clusters. self.cluster_pxl_lst[-1] contains 
      the list of single pixels

    A static method, get_cluster_summary(pixel_clusters), provides summary information

    - n_pixels: number of pixels with energy > 0
    - n_clusters: number of clusters  with >= 2 pixels
    - n_cpixels: number of pixels per cluster
    - circularity: circularity per cluster (0. for linear, 1. for circular)
    - cluster_energies: energy per cluster
    - single_energies: energies in single pixels  
  """
```

``` 
 class miniPIXvis:
  """Analysis of miniPIX frames for low-rate scenarios,
  where on-line analysis is possible and animated graphs are meaningful

    Animated graph of (overlayed) pixel images and cluster properties

    Args:
    - npix: number of pixels per axis (256)
    - nover: number of frames to overlay
    - unit: unit of energy measurement ("keV" or "µs ToT")
    - circ: circularity of "round" clusters (0. - 1.)
    - flat: flatness of energy distribution of pixels in clusters (0. - 1.)
    - acq_time: accumulation time per read-out frame
  """

``` 

Objects of these classes are instantiated by the class `runDAQ`. 
This class also accepts the command-line arguments to set various options, 
as already described above. 


```
  class runDAQ:
    """run miniPIX data acquisition and analysis

    class to handle:

        - command-line arguments
        - event loop controlling data acquisition and data output to file
        - instantiates classes and calls corresponding methods for
          - initialization of miniPIX device
          - real-time analysis of data frames
          - animated figures to show a live view of incoming data
    """
```

Two helper classes implement 1d and 2d histogramming functionality for
efficient and fast animation using methods from `matplotlib.pyplot`.

```
class bhist:
    """one-dimensional histogram for animation, based on bar graph
    supports multiple classes as stacked histogram

    Args:
        * data: tuple of arrays to be histogrammed
        * bindeges: array of bin edges
        * xlabel: label for x-axis
        * ylabel: label for y axis
        * yscale: "lin" or "log" scale
        * labels: labels for classes
        * colors: colors corresponding to labels
    """
```

```
class scatterplot:
    """two-dimensional scatter plot for animation, based on numpy.histogram2d
    supports multiple classes of data, plots a '.' in the corresponding color
    in every non-zero bin of a 2d-histogram

    Args:
        * data: tuple of pairs of cordinates  (([x], [y]), ([], []), ...)
          per class to be shown
        * binedges: 2 arrays of bin edges ([bex], [bey])
        * xlabel: label for x-axis
        * ylabel: label for y axis
        * labels: labels for classes
        * colors: colors corresponding to labels
    """
```

A package script `run_mPIXdaq` is provided as an example to tie 
everything together in a running program. 
Because the ADVACAM *Python* interface (`pypixet.so`) expects 
C-libraries and configuration files in the very same directory 
as *pypixet.so* itself, some tricky manipulation of the environment 
variable `LD_LIBRAREY_PATH` is needed to ensure that all libraries
are loaded and the *miniPIX* is correctly initialized. 

```
#!/usr/bin/env python
#
# script run_mPIXdaq.py
#  run mpixdaq example with data acquisition, on-line analysis and visualization
#  of pixel frames and histogramming

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

```
It is also possible to start the script as a *Python* module:

```
python -m mpixdaq
```
