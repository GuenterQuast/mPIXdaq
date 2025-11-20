"""mPIXdaq: Minimalist Python Script to illustrate data acquisition and data analysis
   for the miniPIX (EDU) device by ADVACAM

Code for reading data from device taken from examples provided by the manufacturer,
see https://wiki.advacam.cz/wiki/Python_API

This code uses standard libraries

  - numpy
  - matplotlib,
  - scipy.ndimage
  - numpy.cov
  - numpy.linalg.eig

to display the pixel energy map, to cluster pixels and to determine
the cluster shapes energies.

This example is meant as a starting point for use of the miniPIX in physics lab courses,
where transparent insights concerning the input data and subsequent analysis steps are
key learning objectives.

"""

import argparse
import sys
import os
import pathlib
import gzip
import time
import yaml
import numpy as np
from queue import Queue
from threading import Thread
from scipy import ndimage

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("dark_background")
from matplotlib.colors import LogNorm


# function for conditional import of ADVACAM libraries
def import_pixet():
    global pypixet
    import platform

    mach = platform.machine()  # machine type
    arch = platform.architecture()  # architecture and  linker format
    syst = platform.system()  # system (Linux, Windows or Darwin
    if mach == 'x86_64':
        from .advacam_x86_64 import pypixet
    elif mach == 'aarch64' and arch[0] == "32bit":
        from .advacam_armhf import pypixet
    elif mach == 'aarch64' and arch[0] == "64bit":
        from .advacam_arm64 import pypixet
    elif syst == "Darwin":
        from .advacam_mac import pypixet
    elif "Windows" in arch[1]:
        if arch[0] == '64bit':
            if sys.version.split()[0] != '3.7.9':
                print("warning - on MS Windows pypixet only works with Python 3.7.9")
            from .advacam_win64 import pypixet
    else:
        exit(" !!! pypixet not available for architecture " + mach + arch[0])


# function for conditional import from npy_append_array !!!
def import_npy_append_array():
    global NpyAppendArray
    from npy_append_array import NpyAppendArray


#
#  main functions and classes - - - - -
#

# - handling the miniPIX device


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

    def __init__(self, ac_count=10, ac_time=0.1, bad_pixels=None):
        """initialize miniPIX device and set up data acquisition"""

        # no device yet
        self.dev = None

        # import python interface to ADVACAM libraries
        try:
            import_pixet()
        except Exception as e:
            print("!!! failed to import pypixet library ", str(e))
            return

        # start miniPIX software
        rc = pypixet.start()
        if rc != 0:
            print("!!! return code from pypixet.start():", rc)
            return
        if not pypixet.isrunning():
            print("!!! pipixet did not start!")
            return
        self.pixet = pypixet.pixet
        print("*==* loaded pypixet vers. ", self.pixet.pixetVersion())
        devs = self.pixet.devicesByType(self.pixet.PX_DEVTYPE_MPX2)  # miniPIX uses the mediPIX 2 chip
        if len(devs) == 0:
            print("!!! no miniPIX device found")
            return
        # retrieve device parameters
        self.id = 0
        self.dev = devs[self.id]
        print("*==* found device " + self.dev.parameters().get("DeviceName").getString())
        self.npx = self.dev.width()
        # options for data acquisition
        # OPMs = ["PX_TPXMODE_MEDIPIX", "PX_TPXMODE_TOT", "PX_TPXMODE_1HIT", "PX_TPXMODE_TIMEPIX"]
        # device initialization
        pixcfg = self.dev.pixCfg()  # Create the pixels configuration object
        # enable output of pixel energies
        pixcfg.setModeAll(self.pixet.PX_TPXMODE_TOT)
        if self.dev.useCalibration(1):  # pixel values in keV
            print("!!! Could not enable device calibration")
        else:
            print("*==* running in ToT mode converted to keV")
        # parameters controlling data acquisition

        #  -  ac_count, ac_time, fileType, fileName
        #     if ac_count>1: frame data is available only from last frame
        self.ac_count = ac_count
        self.ac_time = ac_time
        self.bad_pixels = bad_pixels

        # ring buffer for data collection
        self.Nbuf = 8
        self.fBuffer = np.zeros((self.Nbuf, self.npx * self.npx), dtype=np.float32)
        self._w_idx = 0

        # Queues for communication and synchronization
        #    1. a Queue with less slots than buffers to enforce blocking if no buffer space left
        self.dataQ = Queue(self.Nbuf - 2)
        #    2. a Queue to terminate the process
        self.cmdQ = Queue(1)

    def device_info(self):
        pars = self.dev.parameters()
        dn = pars.get("DeviceName").getString()
        fw = pars.get("Firmware").getString()
        temp = pars.get("Temperature").getDouble()
        bias = pars.get("BiasSense").getDouble()
        frq = self.dev.timepixClock()
        print("miniPIX device info:")
        print(f"   {dn}, Firmware: {fw}")
        print(f"   Temp: {temp:.1f}, Bias: {bias:.1f}, frequency: {frq:.2f} MHz")
        print(
            f"   sensor type: {self.dev.sensorType(self.id)}"
            + f"  pitch: {self.dev.sensorPitch(self.id)} µm"
            + f"  thickness: {self.dev.sensorThickness(self.id)} µm"
            + f"  width {self.npx}  height: {self.dev.height()}"
        )
        rc, n_good, n_bad, frame = self.dev.doDigitalTest()
        print(f"   good pixels {n_good},  bad pixels {n_bad}") if rc == 0 else print("  Digital test failed")
        print(f"   acquisition time min: {self.dev.acqTimeMin()} s    max: {self.dev.acqTimeMax()} s")
        hasCalibration = self.dev.hasCalibration()
        npx = self.npx
        if hasCalibration:
            a = [0] * npx * npx
            b = [0] * npx * npx
            c = [0] * npx * npx
            t = [0] * npx * npx
            self.dev.calibrationDataAbct(self.id, a, b, c, t)
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            t = np.array(t)
            print(
                "   calibration parameters:"
                + f"  a: {a.mean():.3g} +/- {a.std():.2g}"
                + f"  b: {b.mean():.3g} +/- {b.std():.2g}"
                + f"  c: {c.mean():.3g} +/- {c.std():.2g}"
                + f"  t: {t.mean():.3g} +/- {t.std():.2g}"
            )
        else:
            print("   No calibration parameters found on Chip")

    def __call__(self):
        """Read *ac_count* frames with *ac_time* accumulation time each and add all up;
        return pointer to buffer data via Queue
        """
        while self.cmdQ.empty():
            rc = self.dev.doSimpleIntegralAcquisition(self.ac_count, self.ac_time, self.pixet.PX_FTYPE_AUTODETECT, "")
            if rc != 0:
                print("!!! miniPIX device readout error: ", self.dev.lastError())
                self.dataQ.put(None)
            # get frame and store in ring buffer
            self.fBuffer[self._w_idx, :] = np.asarray(self.dev.lastAcqFrameRefInc().data())
            # remove noisy pixels from (linear) data frame based on bad-pixel list
            if self.bad_pixels is not None:
                self.fBuffer[self._w_idx][self.bad_pixels] = -1
            self.dataQ.put(self._w_idx)
            self._w_idx = self._w_idx + 1 if self._w_idx < self.Nbuf - 1 else 0

    def get_linear_framedata(self, r_idx):
        """return frame data in compact ascii format "[[i0, b[i0]], ..., [iN,b[iN]]"

        Args:
          - r_idx: buffer index with frame data of interest

        Returns:
          - stacked array with indices and non-zero values;
            output as ascii via yaml.dump(np.column_stack(idx, b[idx]).tolist(), default_flow_style=True)
        """

        b = self.fBuffer(r_idx)
        idx = np.argwhere(b > 0)
        return np.column_stack(idx, b[idx])

    def __del__(self):
        pypixet.exit()


# - class and functions for data analysis


class frameAnalyzer:
    def __init__(self, csv=None):
        """Analyze frame data and produce a list of cluster objects,

        Args:

           csv: file in text format (csv) to (optionally) write csv header and data

        Output:

          pixel_clusters: a list of tuples of format
            (x, y), n_pix, energy, (var_mx, var_mn), angle, (xEm, yEm), (varE_mx, varE_mn) )
          with cluster properties

          A static method, cluster_summary, calculates a summary of the clusters
          in a pixel frame, returning
            - n_clusters: number of multi-pixel clusters
            - n_cpixels: number of pixels per cluster
            - circularity: circularity per cluster (0. for linear, 1. for circular)
            - flatness:  ratio of maximum variances of pixel and energy distributions in clusters
            - cluster_energies: energy per cluster
            - single_energies: energies in single pixels
        """

        self.csvfile = csv
        if csv is not None:
            self.write_csvheader()

        # set parameters for analysis
        # - structure for connecting pixels in scipy.ndimage.label
        self.label_structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        #  - maximum number of clusters per frame
        self.max_n_clusters = 250
        self.warning_issued = False

        # initialize
        # - an empty numpy array for null results
        # - start-time
        self.t_start = None

        # initialize data members for results
        self.pixel_clusters = []  # result of __call__()
        self.cluster_pxl_lst = []  # result of find_connected()
        self.cluster_summary = None  # result of static method get_cluster_summary(pixel_clusters)

    def covmat_2d(self, x, y, vals):
        """Covariance matrix of a sampled 2d distribution

        Args:
          - x:  x-index _ix of array vals
          - y:  y-index _iy of array vals
          - vals: 2d array vals[_ix, _iy]

        Returns:
          mean: mean_x and mean_y of distribution
          covmat: covariance matrix of distribution

        (this is probably not the most efficient implementation ...)
        """

        xy_x, xy_y = np.meshgrid(x, y)
        _sum = vals[xy_x, xy_y].sum()
        _sumx = np.matmul(vals[xy_x, xy_y], x).sum()
        _sumx2 = np.matmul(vals[xy_x, xy_y], x * x).sum()
        _sumy = np.matmul(y, vals[xy_x, xy_y]).sum()
        _sumy2 = np.matmul(y * y, vals[xy_x, xy_y]).sum()
        _sumxy = np.matmul(y, np.matmul(vals[xy_x, xy_y], x))

        meanx = _sumx / _sum
        varx = _sumx2 / _sum - meanx * meanx
        meany = _sumy / _sum
        vary = _sumy2 / _sum - meany * meany
        cov = _sumxy / _sum - meanx * meany
        #
        mean = np.array([meanx, meany])
        covmat = np.array([[varx, cov], [cov, vary]])
        return (mean, covmat)

    def find_connected(self, f):
        """find connected areas in pixel image using label() from scipy.ndimage

        label() works with binary frame data (0/1 or False/True)

        Args:

          - frame: 2d-frame from the miniPIX

        Returns:

          - pixel_list: list of pixel coordinates for each cluster

            single pixels are collected in last list item (i.e. pixel_list[n_clusters]), where
            n_clusters, the number of clusters with 2 or more pixels,  is len(pixel_list) - 1;
            the list of unclustered pixels is pixel_list[n_clusters], and the number of
            single pixels is len(pixel_list[n_clusters]).
        """

        # initialize results lists, separating clusters fom single hits
        f_labeled, n_labels = ndimage.label(f > 0, structure=self.label_structure)

        # separating clusters fom single hits
        pixel_list = []
        sngl_pxl_list = []
        f_labeled, n_labels = ndimage.label(f > 0, structure=self.label_structure)

        # avoid analyzing unreasonably large number of clusters
        if n_labels > self.max_n_clusters:
            if not self.warning_issued:
                print(f"!!! frameAnalyzer: rejecting frames with > {self.max_n_clusters} clusters !")
                self.warning_issued = True
            return pixel_list

        # retrieve and store cluster data
        for _l in range(1, 1 + n_labels):
            pl = np.argwhere(f_labeled == _l)
            if len(pl) == 1:
                # collect single pixels in one pixel list
                sngl_pxl_list.append(pl[0])
            else:
                pixel_list.append(pl)
        # append single pixels
        pixel_list.append(sngl_pxl_list)

        # alternative clustering using sclearn.cluster.DBSCAN (is a bit slower)
        # px_list = np.argwhere(f > 0)
        # cluster_result = DBSCAN(eps=1.5, min_samples=2, algorithm="ball_tree").fit(px_list)
        # clabels = self.cluster_result.labels_
        # pixel_list = []
        # for _l in set(clabels)):
        #     pl = px_list[clabels == _l]
        #     pixel_list.append(pl)

        return pixel_list

    def cluster_properties(self, f, pl):
        """
        Create a tuples with properties per cluster:
            - mean of x and y coordinates,
            - number of pixels,
            - total energy of cluster,
            - eigenvalues of covariance matrix and orientation (angle in range [-pi/2, pi/2])
            - mean x and y of energy distribution
            - minimal and maximal eigenvalues of the covariance matrix of the energy distribution:

        Args:

            - f: pixel frame
            - pl: list of pixels contributing to cluster

        Returns:
            clusters: tuple of format
            (x, y), n_pix, energy, (var_mx, var_mn), angle, (xEm, yEm), (varE_mx, varE_mn) )

        """

        npix = len(pl)
        if npix > 1:
            # energy in cluster
            energy = f[pl[:, 0], pl[:, 1]].sum()
            #  - mean values of x and y and covariance matrix of pixel area
            x_mean = pl[:, 0].mean(dtype=np.float32)
            y_mean = pl[:, 1].mean(dtype=np.float32)

            # analyze cluster shape
            # - covariance matrix of cluster area
            _evals, _evecs = np.linalg.eigh(np.cov(pl[:, 0], pl[:, 1], dtype=np.float32))
            _idmx = 0 if _evals[0] > _evals[1] else 1
            var_mx, var_mn = _evals[_idmx], _evals[1 - _idmx]
            # orientation of cluster
            angle = np.arctan2(_evecs[_idmx, 0], _evecs[_idmx, 1])
            if angle > np.pi / 2.0:
                angle -= np.pi
            elif angle < -np.pi / 2.0:
                angle += np.pi

            # - covariance matrix of energy distribution
            (xEm, yEm), covmat = self.covmat_2d(pl[:, 0], pl[:, 1], f)
            # calculate eigenvalues and orientation
            _evals, _evecs = np.linalg.eigh(covmat)
            _idmx = 0 if _evals[0] > _evals[1] else 1
            varE_mx, varE_mn = _evals[_idmx], _evals[1 - _idmx]
        else:  # single-pixel object
            _x, _y = pl[0]
            x_mean = _x
            y_mean = _y
            energy = f[_x, _y]
            var_mx, var_mn = (0, 0)
            angle = 0.0
            xEm, yEm = _x, _y
            varE_mx, varE_mn = (0, 0)
        return ((x_mean, y_mean), npix, energy, (var_mx, var_mn), angle, (xEm, yEm), (varE_mx, varE_mn))

    def write_csvheader(self):
        """Write csv header line to file"""
        csvHeader = "time,x_mean,y_mean,n_pix,energy,var_mx,var_mn,angle,xE_mean,yE_mean,varE_mx,varE_mn"
        self.csvfile.write(csvHeader + '\n')

    def write_csv(self, pixel_clusters):
        """Write cluster data to csv file"""
        for _xym, _npix, _energy, _var, _angle, _xyEm, _varE in pixel_clusters:
            if _npix == 0:
                return
            print(
                f"{self.t_frame:.3f}, {_xym[0]:.2f}, {_xym[1]:.2f}, {_npix}, {_energy:.1f}"
                + f", {_var[0]:.2f}, {_var[1]:.2f}, {_angle:.2f}"
                + f", {_xyEm[0]:.2f}, {_xyEm[1]:.2f}, {_varE[0]:.3f}, {_varE[1]:.3f}",
                file=self.csvfile,
            )

    @staticmethod
    def get_cluster_summary(pixel_clusters):
        """summarize cluster properties from cluster tuples of format
              ( (x,y), n_pix, energy, (var_mx, var_mn), angle, (xE, yE), (varE_mx, VarE_mn))

        Args:

          - clusters: list of cluster tuples (as produced by frameAnalyzer.cluster_properties())

        Returns:

          - n_clusters: number of multi-pixel clusters
          - n_cpixels: number of pixels per cluster
          - circularity: circularity per cluster (0. for linear, 1. for circular)
          - flatness:  ratio of maximum variances of pixel and energy distributions in clusters
          - cluster_energies: energy per cluster
          - single_energies: energies in single pixels
        """

        id_npix = 1
        id_e = 2
        id_var = 3
        id_varE = 6

        if pixel_clusters is None:
            return None

        # count muli- and single-pixel clusters
        npix = np.asarray([pixel_clusters[_i][id_npix] for _i in range(len(pixel_clusters))])
        n_multipix = len(npix[npix > 1])
        n_singlepix = len(npix[npix == 1])
        # create arrays to store info
        n_cpixels = np.zeros(n_multipix + 1, dtype=np.int32)
        cluster_energies = np.zeros(n_multipix + 1, dtype=np.float32)
        circularity = np.zeros(n_multipix + 1, dtype=np.float32)
        flatness = np.zeros(n_multipix + 1, dtype=np.float32)
        single_energies = np.zeros(max(1, n_singlepix), dtype=np.int32)
        # collect summary info
        _idx_mlt = 0
        _idx_sngl = 0
        for c in pixel_clusters:
            _npx = c[id_npix]
            if _npx > 1:  # clusters with more than one pixel
                n_cpixels[_idx_mlt] = _npx
                cluster_energies[_idx_mlt] = c[id_e]
                circularity[_idx_mlt] = c[id_var][1] / c[id_var][0]
                flatness[_idx_mlt] = c[id_varE][0] / c[id_var][0]
                _idx_mlt += 1
            else:
                # single pixels
                single_energies[_idx_sngl] = c[id_e]
                _idx_sngl += 1
        # finally, add summary of single-pixel-clusters
        n_cpixels[n_multipix] = n_singlepix
        cluster_energies[n_multipix] = single_energies.sum()
        return n_multipix, n_cpixels, circularity, flatness, cluster_energies, single_energies

    def __call__(self, f):
        """Analyze frame data

          - find clusters using scipy.ndimage.label()
          - compute cluster energies
          - compute position and covariance matrix of x- and y-coordinates
          - compute covariance matrix of the energy distribution
          - analyze cluster shape (using eigenvalues of covariance matrix)
          - construct a tuple with cluster properties
          - optionally write cluster data to a file in csv format

        Note: this algorithm only works if clusters do not overlap!

        Args: a 2d-frame from the miniPIX

        Returns:

          - self.pixel_clusters: list of tuples with properties per cluster: mean of x and y
            coordinates, the number of pixels, energy, eigenvalues of covariance matrix and
            their orientation as an angle in range [-pi/2, pi/2] and the minimal and maximal
            eigenvalues of the covariance matrix of the energy distribution. The format is:
            ( (x, y), n_pix, energy, (var_mx, var_mn), angle, (xEm, yEm), (varE_mx, varE_mn) )

          - self.cluster_pxl_lst is a list of dimension n_clusters + 1 and contains the pixel indices
            contributing to each of the clusters. self.cluster_pxl_lst[-1] contains the list of single pixels

        A static method, get_cluster_summary(pixel_clusters), provides summary infomation

          - n_pixels: number of pixels with energy > 0
          - n_clusters: number of clusters  with >= 2 pixels
          - n_cpixels: number of pixels per cluster
          - circularity: circularity per cluster (0. for linear, 1. for circular)
          - cluster_energies: energy per cluster
          - single_energies: energies in single pixels
        """

        # clear output while processing new data
        self.pixel_clusters = []

        # find clusters (lines,  circular  and unassigned = single pixels)
        self.n_pixels = (f > 0).sum()
        if self.n_pixels == 0:
            return None

        # timing
        if self.t_start is None:
            self.t_start = time.time()
        self.t_frame = time.time() - self.t_start

        # find connected pixel areas ("clusters") in frame
        self.cluster_pxl_lst = self.find_connected(f)
        n_objects = len(self.cluster_pxl_lst)
        if n_objects == 0:
            return None

        self.n_clusters = n_objects - 1
        self.n_single = len(self.cluster_pxl_lst[-1])

        # - crate list of properties per cluster in format
        #      ( (x,y), n_pix, energy, (var_mx, var_mn), angle, (xE, yE), (varE_mx, VarE_mn))
        # loop over clusters ...
        for _i in range(self.n_clusters):
            # number of pixels in cluster
            pl = self.cluster_pxl_lst[_i]
            self.pixel_clusters.append(self.cluster_properties(f, pl))
        # and single pixels
        for _is in range(self.n_single):
            self.pixel_clusters.append(self.cluster_properties(f, [self.cluster_pxl_lst[self.n_clusters][_is]]))

        #     calculate summary of clusters in frame
        #        self.cluster_summary = self.get_cluster_summary(self.pixel_clusters)
        return self.pixel_clusters


class miniPIXvis:
    """display of miniPIX frames and histograms for low-rate scenarios
    where on-line analysis is possible and animated graphs are meaningful

    Animated graph of (overlayed) pixel images, number of clusters per frame and histograms of cluster properties
    """

    def on_mpl_close(self, event):
        """call-back for matplotlib 'close_event'"""
        self.mpl_active = False

    def __init__(self, npix=256, nover=10, unit='keV', circ=0.5, flat=0.5, acq_time=1.0, badpixels=None):
        """initialize figure with pixel image, rate history and two histograms and a scatter plot

        Args:
           - npix: number of pixels per axis (256)
           - nover: number of frames to overlay
           - unit: unit of energy measurement ("keV" or "µs ToT")
           - circ: circularity of "round" clusters (0. - 1.)
           - flat: flatness of enery distrbituion of pixels in clusters (0. - 1.)
           - acq_time: accumulation time per read-out frame
        """

        self.npx = npix
        self.n_overlay = nover
        self.circularity_cut = circ
        self.flatness_cut = flat
        self.unit = unit
        self.acq_time = acq_time

        # maximum number of frames in scatter plot
        self.max_n_frames_for_scatter_plot = 5000
        self.warning_issued = False

        # - data structure to store miniPIX frames and analysis results per frame
        self.framebuf = np.zeros((self.n_overlay, self.npx, self.npx), dtype=np.float32)
        self.i_buf = 0
        self.i_frame = 0
        # cumulative image
        self.cimage = np.zeros((self.npx, self.npx), dtype=np.float32)
        # frame summary statistics
        self.Energy = 0.0
        self.np_unass = 0
        self.E_unass = 0.0
        self.N_clusters = 0

        # variables for cluster statistics and histograms
        self.n_clusters = 0
        self.n_cpixels = np.array([])
        self.circularity = np.array([])
        self.flatness = np.array([])
        self.cluster_energies = np.array([])
        self.single_energies = np.array([])

        # - prepare a figure with subplots
        self.fig = plt.figure('PIX data', figsize=(12.0, 8.5), facecolor="#1f1f1f")
        self.fig.suptitle("miniPiX Data Acquisition and Analysis", size="xx-large", color="cornsilk")
        self.fig.subplots_adjust(left=0.05, bottom=0.03, right=0.99, top=0.99, wspace=0.30, hspace=0.1)
        nrows = 16
        col1, col2, col3 = 20, 24, 30
        ncols = col3 + 1
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols)
        plt.tight_layout()
        self.fig.canvas.mpl_connect('close_event', self.on_mpl_close)
        self.mpl_active = True

        # - - 2D display for pixel map
        #  bad-pixel map for hanling of bad pixels
        if badpixels is None:
            self.badpixel_map = None
        else:  # create bad-pixel map as masked array
            bp = np.zeros(self.npx * self.npx)
            bp[badpixels] = 1.0
            badpixel_map = np.ma.masked_where(bp == 1, bp).reshape((self.npx, self.npx))
            # badpixel_map = bp.reshape((self.npx, self.npx))
        # pixel image
        self.axim = self.fig.add_subplot(gs[:, :col1])
        self.axim.set_title("Pixel Energy Map " + self.unit, y=0.975, size="x-large")
        self.axim.set_xlabel("# x        ", loc="right")
        self.axim.set_ylabel("# y             ", loc="top")
        # no default frame around graph, but show detector boundary
        self.axim.set_frame_on(False)

        if badpixels is not None:
            _ = self.axim.imshow(badpixel_map, origin="lower", cmap='gray', vmax=10.0)
        self.vmin, vmax = 0.5, 500
        self.img = self.axim.imshow(np.zeros((self.npx, self.npx)), origin="lower", cmap='hot', norm=LogNorm(vmin=self.vmin, vmax=vmax))
        cbar = self.fig.colorbar(self.img, shrink=0.6, aspect=40, pad=-0.0375)
        self.img.set_clim(vmin=self.vmin, vmax=vmax)
        # cbar.set_label("Energy " + unit, loc="top", labelpad=-5 )
        if self.acq_time is not None and self.acq_time > 0.0:
            txt_overlay = f"integration time {acq_time * nover:.1f} s"
        else:
            txt_overlay = f"overlay of {int(self.n_overlay)} frames"
        self.axim.text(0.01, -0.06, txt_overlay, transform=self.axim.transAxes, color="royalblue")
        self.im_text = self.axim.text(0.02, -0.085, "#", transform=self.axim.transAxes, color="r", alpha=0.75)
        # detector geometry
        col_geom = "gray"
        _rect = mpl.patches.Rectangle((0, 0), self.npx, self.npx, linewidth=1, edgecolor=col_geom, facecolor='none')
        self.axim.add_patch(_rect)
        # blue arrow showing detector dimension in mm
        self.axim.arrow(146, 261.0, 110.0, 0, length_includes_head=True, width=1.5, color=col_geom)
        self.axim.arrow(110, 261.0, -110.0, 0, length_includes_head=True, width=1.5, color=col_geom)
        self.axim.text(115.0, 259, "14 mm")
        # 2nd x-axis in mm
        pitch = 0.055  # pixel size
        px2x = lambda x: x * pitch
        x2px = lambda x: x / pitch
        axim_x2 = self.axim.secondary_xaxis(0.935, functions=(px2x, x2px))
        axim_x2.set_frame_on(False)
        axim_x2.set_xlabel('Position [mm]', loc='right', color=col_geom)
        axim_x2.set_xlim((0.0, 14.08))
        axim_x2.tick_params(colors=col_geom)

        # a (vertical) rate display
        self.axRate = self.fig.add_subplot(gs[2 : nrows - 1, col1 : col2 - 1])
        pos = self.axRate.get_position()
        self.axRate.set_position([0.985 * pos.x0, pos.y0, pos.width, pos.height])
        self.axRate.xaxis.set_label_position('top')
        # self.axRate.set_ylabel('History [frame #]', rotation=-90, labelpad=15.0)
        self.axRate.set_xlabel('objects/frame')
        self.axRate.grid(linestyle='dotted', which='both')
        self.num_history_points = 300
        self.axRate.set_ylim(-self.num_history_points, 0.0)
        self.hrates = self.num_history_points * [None]
        _yplt = np.linspace(-self.num_history_points, 0.0, self.num_history_points)
        (self.line_rate,) = self.axRate.plot(self.hrates, _yplt, '.--', lw=1, markersize=4, color="#F0F0FC", mec="orange")
        self.line_avrate = self.axRate.axvline(0.0, linestyle='--', lw=1, color="red")
        self.rate_mx = 5
        self.axRate.set_xlim(-0.25, self.rate_mx)

        #  - histogram of pixel energies
        self.axh1 = self.fig.add_subplot(gs[1:5, col2:])
        nbins1 = 45
        min1 = 5
        max1 = 1300
        # be1 = np.linspace(min1, max1, nbins1 + 1, endpoint=True)
        # log scale
        be1 = np.geomspace(min1, max1, nbins1 + 1, endpoint=True)
        self.bhist1 = bhist(
            ax=self.axh1,
            data=([],),
            binedges=be1,
            xlabel="pixel energies " + self.unit,
            ylabel="",
            xscale="log",
            yscale="log",
            labels=None,
            colors=('r',),
        )

        # - histogram of cluster energies
        self.axh2 = self.fig.add_subplot(gs[6:10, col2:])
        nbins2 = 50
        min2 = 5  # 5keV
        max2 = 11999  # 12.5 MeV
        # be2 = np.linspace(min2, max2, nbins2 + 1, endpoint=True)
        be2 = np.geomspace(min2, max2, nbins2 + 1, endpoint=True)
        self.bhist2 = bhist(
            ax=self.axh2,
            data=([], [], []),
            binedges=be2,
            xlabel="cluster energies " + self.unit,
            ylabel="",
            xscale="log",
            yscale="log",
            labels=("linear (β)", "circular (α)", "singles"),
            colors=('yellow', 'cyan', 'red'),
        )

        # - scatter plot: cluster energies & sizes
        self.ax3 = self.fig.add_subplot(gs[11:15, col2:])
        mxx = 11999
        bex = np.linspace(0.0, mxx, 300, endpoint=True)
        mxy = 55
        bey = np.linspace(0.0, mxy, mxy, endpoint=True)
        # initialize for 3 classes of ([x],[y]) pairs
        self.scpl = scatterplot(
            ax=self.ax3,
            data=(([], []), ([], []), ([], [])),
            binedges=(bex, bey),
            xlabel="cluster energies (keV)",
            ylabel="pixels per cluster",
            labels=("linear (β)", "circular (α)", "single pixels"),
            colors=('yellow', 'cyan', 'r'),
        )

        # show plots in interactive mode
        plt.ion()
        plt.show()

        self.dt_last_plot = 0.0
        self.t_start = time.time()

    def upd_histograms(self, frame2d, cluster_summary):
        """update histograms"""

        n_clusters, n_cpixels, circularity, flatness, cluster_energies, single_energies = cluster_summary

        if n_clusters > 0:
            self.Energy = cluster_energies.sum()
            self.Energy_in_clusters = cluster_energies[:n_clusters].sum()
            self.np_unass = n_cpixels[n_clusters]  # last entry is for unassigned
            self.E_unass = cluster_energies[n_clusters]
        else:
            self.Energy = frame2d[frame2d > 0].sum()  # raw pixel energy
            self.Energy_in_clusters = 0.0
            self.np_unass = len(frame2d[frame2d > 0])
            self.E_unass = self.Energy
        self.N_clusters = n_clusters

        # boolean indices for linear and circular, spiky objects
        is_round = circularity[:n_clusters] > self.circularity_cut
        is_flat = flatness[:n_clusters] > self.flatness_cut

        is_alpha = is_round & ~is_flat

        # update histogram 1 with pixel energies
        self.bhist1.add((frame2d[frame2d > 0],))

        # update histogram 2 with cluster energies
        if n_clusters > 0:
            self.bhist2.add((cluster_energies[:n_clusters][~is_alpha], cluster_energies[:n_clusters][is_alpha], single_energies))

        # update scatter
        #    protect because of large memory need of scatter plot
        if self.i_frame > self.max_n_frames_for_scatter_plot:
            if not self.warning_issued:
                print(f"!!! anaviz: stop updating scatter plot due to large number of frames")
                self.warning_issued = True
            return
        xlin = cluster_energies[:n_clusters][~is_alpha]
        ylin = n_cpixels[:n_clusters][~is_alpha]
        xcir = cluster_energies[:n_clusters][is_alpha]
        ycir = n_cpixels[:n_clusters][is_alpha]
        self.scpl.add([(xlin, ylin), (xcir, ycir), ([self.E_unass], [self.np_unass])])

    def __call__(self, frame2d, cluster_summary, dt_alive):
        """update cumulative pixel image and rate, analyze data
        and update histograms, scatter plot and status text
        """
        self.i_frame += 1
        # subtract oldest frame ...
        self.cimage -= self.framebuf[self.i_buf]
        # ... and store new one in ring-buffer
        self.framebuf[self.i_buf, :, :] = frame2d[:, :]
        # and add actual data to cumulated values
        self.cimage += frame2d
        # add resp. concatenate information on cluster properties
        if cluster_summary is not None:
            n_clusters, n_cpixels, circularity, flatness, cluster_energies, single_energies = cluster_summary
            n_objects = n_clusters + n_cpixels[n_clusters]
            self.n_clusters += n_clusters
            self.n_cpixels = np.concatenate((self.n_cpixels, n_cpixels))
            self.circularity = np.concatenate((self.circularity, circularity))
            self.flatness = np.concatenate((self.flatness, flatness))
            self.cluster_energies = np.concatenate((self.cluster_energies, cluster_energies))
            self.single_energies = np.concatenate((self.single_energies, single_energies))
        else:
            n_objects = 0
            n_clusters = 0

        # update rate plot
        if n_objects > self.rate_mx:
            self.rate_mx = n_objects
            self.axRate.set_xlim(0.25, 1.05 * self.rate_mx)
        self.hrates[(self.i_frame - 1) % self.num_history_points] = np.float32(n_objects)
        k = self.i_frame % self.num_history_points
        self.line_rate.set_xdata(np.concatenate((self.hrates[k + 1 :], self.hrates[: k + 1])))
        # self.axRate.relim()
        # self.axRate.autoscale_view()
        _n = min(self.num_history_points, self.i_frame)
        self.line_avrate.set_xdata([np.asarray(self.hrates)[:_n].mean()])

        if self.i_buf < self.n_overlay - 1:
            self.i_buf += 1
        else:
            # buffer filled, visualize data
            summary = (self.n_clusters, self.n_cpixels, self.circularity, self.flatness, self.cluster_energies, self.single_energies)
            self.upd_histograms(self.cimage, summary)
            # reset buffer index and cumulative variables
            self.i_buf = 0
            self.n_clusters = 0
            self.n_cpixels = np.array([])
            self.circularity = np.array([])
            self.flatness = np.array([])
            self.cluster_energies = np.array([])
            self.single_energies = np.array([])

        dt_active = time.time() - self.t_start
        # update, redraw and show all subplots in figure
        if dt_active - self.dt_last_plot > 0.08:  # limit number of graphics updates
            # dead_time_fraction = 1.0 - dt_alive / dt_active
            status = (
                f"#{self.i_frame}   active {dt_active:.0f}s   alive {dt_alive:.0f}s "
                + f"  clusters = {self.N_clusters:.0f} / {self.Energy:.0f}keV "
                + f"  single pixels: {self.np_unass:.0f} / {self.E_unass:.0f}keV"
                + 10 * " "
            )
            self.img.set(data=self.cimage)
            self.im_text.set_text(status)
            # self.fig.canvas.start_event_loop(0.001)  # better than plt.pause(), which would steal the focus
            self.fig.canvas.update()
            self.fig.canvas.flush_events()
            self.dt_last_plot = dt_active


# helper classes and functions  - - - - -
class bhist:
    """one-dimensional histogram for animation, based on bar graph
    supports multiple classes as stacked histogram

    Args:
        * data: tuple of arrays to be histogrammed
        * bindeges: array of bin edges
        * xlabel: label for x-axis
        * ylabel: label for y axis
        * xscale: "linear" or "log" x-scale
        * yscale: "linear" or "log" y-scale
        * labels: labels for classes
        * colors: colors corresponding to labels
    """

    def __init__(self, ax=None, data=None, binedges=None, xlabel="x", ylabel="freqeuency", yscale="log", xscale="linear", labels=None, colors=None):
        # ### own implementation of one-dimensional histogram (numpy + pyplot bar) ###

        if type(data) != type((1,)):
            print("! bhist requires a tuple as input, not ", type(data))

        self.n_classes = len(data)

        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot()
        else:
            self.ax = ax

        if labels is None:
            if self.n_classes == 1:
                labels = [None]
            else:
                labels = ["class " + str(_ic) for _ic in range(self.n_classes)]
        if colors is None:
            colors = self.n_classes * [None]

        self.bheights = []
        self.bars = []
        # plot class 1
        _bc, self.be = np.histogram(data[0], binedges)  # histogram data
        self.bheights.append(_bc)
        self.bcnt = (self.be[:-1] + self.be[1:]) / 2.0
        self.w = 0.8 * (self.be[1:] - self.be[:-1])
        ec = 'gray'
        self.bars.append(plt.bar(self.bcnt, _bc, align="center", width=self.w, color=colors[0], label=labels[0], edgecolor=ec, alpha=0.75))
        sum = _bc

        # plot other classes
        for _ic in range(1, self.n_classes):
            _bc, _be = np.histogram(data[_ic], binedges)  # histogram data
            self.bheights.append(_bc)

            self.bars.append(
                plt.bar(self.bcnt, _bc, align="center", width=self.w, color=colors[_ic], label=labels[_ic], edgecolor=ec, alpha=0.75, bottom=sum)
            )
            sum = sum + _bc

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        _mx = max(sum)
        self.maxh = _mx if _mx > 0.0 else 1000
        self.ax.set_ylim(0.75, self.maxh)
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)
        if labels[0] is not None:
            self.ax.legend(loc="upper right")

    def set(self, data):
        """set new histogram data

        Args:
           * data: heights for each bar

        Action: update pyplot bar graph
        """

        sum = np.zeros(len(self.bheights[0]))
        for _i in range(self.n_classes):
            _ic = self.n_classes - 1 - _i
            _bc, _be = np.histogram(data[_ic], self.be)  # histogram data ...
            self.bheights[_ic] = _bc
            for _b, _h in zip(self.bars[_ic], self.bheights[_ic] + sum):
                _b.set_height(_h)
            sum = sum + self.bheights[_ic]

        _mx = max(sum)
        if _mx > self.maxh:
            self.maxh = 1.2 * _mx
            self.ax.set_ylim(0.9, self.maxh)

    def add(self, data):
        """update histogram data

        Args:
            * data: heights for each bar

        Action: update pyplot bar objects
        """

        sum = np.zeros(len(self.bheights[0]))
        for _i in range(self.n_classes):
            # plot bars in reverse order
            _ic = self.n_classes - 1 - _i
            _bc, _be = np.histogram(data[_ic], self.be)  # histogram data ...
            self.bheights[_ic] = self.bheights[_ic] + _bc  # and add to existing
            for _b, _h in zip(self.bars[_ic], self.bheights[_ic] + sum):
                _b.set_height(_h)
            sum = sum + self.bheights[_ic]

        _mx = max(sum)
        if _mx > self.maxh:
            self.maxh = 1.2 * _mx
            self.ax.set_ylim(0.9, self.maxh)


class scatterplot:
    """two-dimensional scatter plot for animation, based on numpy.histogram2d
    supports multiple classes of data, plots a '.' in the corresponding color
    in every non-zero bin of a 2d-histogram

    Args:
        * data: tuple of pairs of cordinates (([x], [y]), ([], []), ...)
          per class to be shown
        * binedges: 2 arrays of bin edges [[bex], [bey]]
        * xlabel: label for x-axis
        * ylabel: label for y axis
        * labels: labels for classes
        * colors: colors corresponding to labels
    """

    def __init__(self, ax=None, data=None, binedges=None, xlabel="x", ylabel="y", labels=None, colors=None):
        #  own implementation of 2d scatter plot (numpy + pyplot.plot() ###

        if type(data) != type((1,)):
            print("! scatterplot requires a tuple as input, not", type(data))

        self.n_classes = len(data)
        # initialize bins
        self.binedges = binedges
        self.bex = binedges[0]
        self.bey = binedges[1]
        self.bcntx = (self.bex[:-1] + self.bex[1:]) / 2.0
        self.bcnty = (self.bey[:-1] + self.bey[1:]) / 2.0
        # bin widths
        self.bwx = self.bex[1] - self.bex[0]
        self.bwy = self.bey[1] - self.bey[0]
        # fraction of bin width as plot off-set for classes
        self.pofx = [(_i + 1) * self.bwx / (self.n_classes + 1) - self.bwx / 2.0 for _i in range(self.n_classes)]
        self.pofy = [(_i + 1) * self.bwy / (self.n_classes + 1) - self.bwy / 2.0 for _i in range(self.n_classes)]

        self.H2d = []
        for _ic in range(self.n_classes):
            # use numpy histogram2d to creade histogram arrays, one per class
            _H2d, _bex, _bey = np.histogram2d(data[_ic][0], data[_ic][1], self.binedges)
            self.H2d.append(_H2d)

        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot()
        else:
            self.ax = ax
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        # self.ax.set_facecolor('k')

        # create initial plot
        if labels is None:
            if self.n_classes == 1:
                labels = [None]
            else:
                labels = ["class " + str(_ic) for _ic in range(self.n_classes)]
        if colors is None:
            colors = n_classes * [None]

        self.gr = []
        for _ic in range(self.n_classes):
            # _xy_list = np.argwhere(self.H2d[_ic] > 0)
            # _x = self.bcntx[_xy_list[:, 0]]
            # _y = self.bcntx[_xy_list[:, 1]]
            _xidx, _yidx = np.nonzero(self.H2d[_ic])
            _x = self.bcntx[_xidx]
            _y = self.bcnty[_yidx]
            (_gr,) = ax.plot(_x, _y, label=labels[_ic], color=colors[_ic], marker='o', markersize=0.5, ls='', alpha=0.75)
            self.gr.append(_gr)
        self.ax.set_xlim(self.bex[0], self.bex[-1])
        self.ax.set_ylim(self.bey[0], self.bey[-1])
        if labels[0] is not None:
            self.ax.legend(loc="upper right")

    def set(self, data):
        for _ic in range(self.n_classes):
            _H2d, _bex, _bey = np.histogram2d(data[_ic][0], data[_ic][1], self.binedges)  # numpy 2d histogram function
            self.H2d[_ic] = _H2d
            _xidx, _yidx = np.nonzero(self.H2d[_ic])
            self.gr[_ic].set(data=(self.bcntx[_xidx], self.bcnty[_yidx]))

    def add(self, data):
        """update scatter-plot data

        Args:
            * data: new (xy)-paris to be added

        Action: update pyplot line objects
        """

        for _ic in range(self.n_classes):
            _H2d, _bex, _bey = np.histogram2d(data[_ic][0], data[_ic][1], self.binedges)  # numpy 2d histogram function
            self.H2d[_ic] = self.H2d[_ic] + _H2d
            _xidx, _yidx = np.nonzero(self.H2d[_ic])
            self.gr[_ic].set(data=(self.bcntx[_xidx] + self.pofx[_ic], self.bcnty[_yidx] + self.pofy[_ic]))


class runDAQ:
    """run miniPIX data acquisition, analysis and real-time graphics

    class to handle:

        - command-line arguments
        - initialization of miniPIX device of input file
        - real-time analysis of data frames
        - animated figures to show a live view of incoming data
        - event loop controlling data acquisition, data output to file
          graphical display
    """

    def __init__(self, wd_path=None):
        """initialize:

        - options from command line arguments
        - miniPIX detector or optionally input from file
        - graphics display
        """

        # write current directory if no path given
        if wd_path is None:
            #    wd_path = os.getenv("HOME")
            wd_path = os.getcwd()
        self.wd_path = wd_path

        # parse command line arguments
        parser = argparse.ArgumentParser(description="read, analyze, display and histogram data from miniPIX device")
        parser.add_argument('-v', '--verbosity', type=int, default=1, help='verbosity level (1)')
        parser.add_argument('-o', '--overlay', type=int, default=10, help='number of frames to overlay in graph (10)')
        parser.add_argument('-a', '--acq_time', type=float, default=0.1, help='acquisition time/frame (0.1)')
        parser.add_argument('-c', '--acq_count', type=int, default=5, help='number of frames to add (5)')
        parser.add_argument('-f', '--file', type=str, default='', help='file to store frame data')
        parser.add_argument('-w', '--writefile', type=str, default='', help='csv file to write cluster data')
        parser.add_argument('-t', '--time', type=int, default=36000, help='run time in seconds (36000)')
        parser.add_argument('--circularity_cut', type=float, default=0.5, help='cut on cicrularity for alpha detetion')
        parser.add_argument('--flatness_cut', type=float, default=0.4, help='cut on flatness for alpha detection')

        parser.add_argument('-r', '--readfile', type=str, default='', help='file to read frame data')
        parser.add_argument('-b', '--badpixels', type=str, default='', help='file with bad pixels')
        args = parser.parse_args()
        timestamp = time.strftime('%y%m%d-%H%M', time.localtime())

        # set options
        self.verbosity = args.verbosity
        self.out_filename = args.file + '_' + timestamp + '.npy' if args.file != '' else None
        self.read_filename = args.readfile if args.readfile != '' else None
        self.csv_filename = args.writefile if args.writefile != '' else None
        self.fname_badpixels = args.badpixels
        self.acq_time = args.acq_time
        self.acq_count = args.acq_count
        self.n_overlay = args.overlay
        self.circularity_cut = args.circularity_cut
        self.flatness_cut = args.flatness_cut
        self.run_time = args.time

        if self.verbosity > 0:
            print(f"\n*==* script {sys.argv[0]} executing in working directory {self.wd_path}")

        if self.out_filename is not None:
            # data recording with npy_append_array()
            import_npy_append_array()

        # handling of bad pixels
        badpixel_list = None
        if self.fname_badpixels == '':
            fname = "badpixels.txt"  # check default bad-pixel file
            if self.read_filename is None:
                try:
                    badpixel_list = np.loadtxt(fname, dtype=np.int32).tolist()
                    print("*==* list of bad pixels from file ", fname)
                except FileNotFoundError:
                    pass
        else:
            badpixel_list = np.loadtxt(self.fname_badpixels, dtype=np.int32).tolist()
            print("*==* list of bad pixels from file ", self.fname_badpixels)
        # print(self.badpixel_list)

        # try to load pypixet library and connect to miniPIX
        if self.read_filename is None:
            self.tot_acq_time = self.acq_count * self.acq_time
            # initialize data acquisition object
            self.daq = miniPIXdaq(self.acq_count, self.acq_time, bad_pixels=badpixel_list)
            if self.daq.dev is None:
                _a = input("  Problem with miniPIX device - read data from file ? (y/n) > ")
                if _a in {'y', 'Y', 'j', 'J'}:
                    path = os.path.dirname(os.path.realpath(__file__)) + '/'
                    self.read_filename = path + "data/BlackForestStone.npy.gz"
                    badpixel_list = None
                else:
                    exit("Exiting")
            else:  # library and device are ok
                if self.verbosity > 0:
                    print(f"     * overlaying {self.n_overlay} frames with {self.tot_acq_time} s")
                    print(f"     * readout {self.acq_count} x {self.acq_time} s")
                if self.verbosity > 1:
                    self.show_DeviceInfo = True
                    self.daq.device_info()
                self.npx = self.daq.npx
                self.unit = "(keV)" if self.daq.dev.isUsingCalibration() else "ToT (µs)"

        #  end device initialization ---

        # set path to working directory where all output goes
        os.chdir(self.wd_path)

        if self.read_filename is not None:
            # read from file if requested
            if self.verbosity > 0:
                print("*==* data from file " + self.read_filename)
            suffix = pathlib.Path(self.read_filename).suffix
            if suffix == ".gz":
                f = gzip.GzipFile(self.read_filename, mode='r')
                self.fdata = np.load(f)
            elif suffix == ".npy":
                self.fdata = np.load(self.read_filename, mmap_mode="r")
            else:
                exit(" Exit - unknown file extension " + suffix)
            # assume data is 256x256 pixels in keV per pixel
            shape = self.fdata.shape
            if len(shape) < 3 or shape[1] != 256:
                exit(f"unexpected shape {shape} of array, expected 256x256")
            elif shape[1] != 256:
                exit(f"unexpected shape {shape} of array, expected 256x256")
            self.n_frames_in_file = shape[0]
            self.npx = shape[1]
            self.unit = "(keV)"
            self.acq_time = 0.0  # acquitition time unknown, as not stored in file
            self.tot_acq_time = self.acq_count * self.acq_time

            if self.verbosity > 0:
                print(f" found {self.n_frames_in_file} pixel frames in file")

        self.csvfile = None
        if self.csv_filename is not None:
            fn = self.csv_filename + ".csv"
            self.csvfile = open(fn, "w", buffering=100)
            if self.verbosity > 0:
                print("*==* writing clusters to file " + fn)

        # set-up frame analyzer
        self.frameAna = frameAnalyzer(csv=self.csvfile)

        # finally, initialize visualizer
        self.mpixvis = miniPIXvis(
            npix=self.npx,
            nover=self.n_overlay,
            unit=self.unit,
            circ=self.circularity_cut,
            acq_time=self.tot_acq_time,
            badpixels=badpixel_list,
        )

        # keyboard control via thread
        self.ACTIVE = True
        self.kbdQ = Queue()  # Queue for command input from keyboard
        self.kbdthread = Thread(name="kbdInput", target=self.keyboard_input, args=(self.kbdQ,)).start()

    def keyboard_input(self, cmd_queue):
        """Read keyboard input and send to Queue, runing as background-thread to avoid blocking"""
        while self.ACTIVE:
            cmd_queue.put(input())

    def __call__(self):
        """run daq loop"""

        # set up daq
        dt_alive = 0.0
        dt_active = 0.0
        frame2d = np.zeros((self.npx, self.npx), dtype=np.float32)
        i_frame = 0
        # start daq as a Thread
        if self.read_filename is None:
            Thread(target=self.daq, daemon=True).start()

        # start daq loop
        t_start = time.time()
        print("\n" + 15 * ' ' + "\033[36m type 'E<ret>' or close graphics window to end" + "\033[31m", end='\r')
        try:
            while (dt_active < self.run_time) and self.mpixvis.mpl_active and self.ACTIVE:
                if self.read_filename is None:
                    _idx = self.daq.dataQ.get()
                    # data as 2d pixel array
                    frame2d[:, :] = self.daq.fBuffer[_idx].reshape(self.npx, self.npx)
                    dt_alive += self.acq_count * self.acq_time
                    i_frame += 1
                else:  # from file
                    i_frame += 1
                    if i_frame > self.n_frames_in_file:
                        print("\033[36m\n" + 20 * ' ' + "'end-of-file - type <ret> to terminate")
                        break
                    frame2d = self.fdata[i_frame - 1]
                    ##!time.sleep(1.0)
                    time.sleep(0.2)

                # write frame to file ?
                if self.out_filename is not None:
                    with NpyAppendArray(self.out_filename) as npa:
                        npa.append(np.array([frame2d]))

                # analyze frame and retrieve result
                pixel_clusters = self.frameAna(frame2d)
                if pixel_clusters is not None:
                    if self.csvfile is not None:
                        self.frameAna.write_csv(pixel_clusters)
                # animated visualization
                cluster_summary = frameAnalyzer.get_cluster_summary(pixel_clusters)
                self.mpixvis(frame2d, cluster_summary, dt_alive)

                if not self.kbdQ.empty():
                    if self.kbdQ.get() == 'E':
                        self.ACTIVE = False
                #    # heart-beat for console
                dt_active = time.time() - t_start
                print(f"  #{i_frame}  {dt_active:.0f}s", end="\r")

        except KeyboardInterrupt:
            print("\n keyboard interrupt ")
        except Exception as e:
            print("\n excpetion in daq loop: ", str(e))

        finally:
            # end daq loop, print reason for end and clean up
            if not self.ACTIVE:
                print("\033[36m\n" + 20 * ' ' + "'E'nd command received", end='')
            elif not self.mpixvis.mpl_active:
                print("\033[36m\n" + 20 * ' ' + " Graphics window closed", end='')
            elif dt_active > self.run_time:
                print("\033[36m\n" + 20 * ' ' + f"end after {dt_active:.1f} s", end='')

            self.ACTIVE = False
            if self.read_filename is None:
                self.daq.cmdQ.put("e")
            if self.csvfile is not None:
                self.csvfile.flush()
                self.csvfile.close()
            if self.read_filename is None:
                pypixet.exit()
            print(10 * ' ' + "  - type <ret> to terminate ->> ", end='')


if __name__ == "__main__":  # -  - - - - - - - - - -
    rD = runDAQ()
    rD()
