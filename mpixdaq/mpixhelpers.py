"""Helper functions for mpixdaq

- class fileDecoders to decode  various input file formats: mPIXdaq .npy and .yml and Advacam .txt and .clog
- class clusterReader() to read (and analyze) pixel clusters written with mPIXdaq
- function plot_cluster() to plot energy map of pixel cluster
- class shmManager for management of shared memory blocks

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from multiprocessing import shared_memory
import pandas as pd
from scipy.stats import sigmaclip
import re
import yaml
import gzip
import sys


class fileDecoders:
    """Collection of decoders for various input file formats
    supports mPIXdaq .npy and .yml and Advacam .txt and .clog
    """

    @classmethod
    def mPIXdaq_yml(cls, ymlfile):
        """Read data from yaml file (the default file format of mPIXdaq)
        and yield individual frames from file

        Args:
        * ymlfile:  file handle file in mPIXdaq .yml format
        """
        dtype = "unknown"
        meta_blk = ''
        while True:
            _l = ymlfile.readline()
            if not _l:
                break
            if isinstance(_l, bytes):
                _l = _l.decode()  # needed for gzip returning bytes objects
            if _l.startswith("frame_data:"):
                dtype = "frame"
                break
            if _l.startswith("cluster_data"):
                dtype = "clusters"
                break
            meta_blk += _l
        # decode meta-data
        meta_data = yaml.load(meta_blk, Loader=yaml.CSafeLoader)

        if dtype == "frame":
            return meta_data, cls.frame_generator(ymlfile)
        elif dtype == "clusters":
            return meta_data, cls.frame_from_clusters_generator(ymlfile)

    @staticmethod
    def frame_generator(ymlfile):
        """generator returning frames from data block of yaml file with frames"""
        in_datablk = True
        while in_datablk:
            data_blk = ''
            while True:
                _l = ymlfile.readline()
                if not _l:
                    in_datablk = False
                    break
                if isinstance(_l, bytes):
                    _l = _l.decode()  # needed for gzip returning bytes objects
                if _l == '\n':
                    break
                elif _l.startswith("...") or _l.startswith("eor_data:"):
                    in_datablk = False
                    break
                data_blk += _l
            if not in_datablk:
                break
            yield yaml.load(data_blk, Loader=yaml.CSafeLoader)[0]

    @staticmethod
    def frame_from_clusters_generator(ymlfile):
        """generator returning frames from data block of yaml file with clusters"""
        in_datablk = True
        t_stamp0 = 0
        fdata = []
        while in_datablk:
            data_blk = ''
            while True:
                _l = ymlfile.readline()
                if not _l:
                    yield fdata  # deliver last frame
                    in_datablk = False
                    break
                if isinstance(_l, bytes):
                    _l = _l.decode()  # needed for gzip returning bytes objects
                if _l == '\n':
                    break
                elif _l.startswith("...") or _l.startswith("eor_data:"):
                    yield fdata  # deliver last frame
                    in_datablk = False
                    break
                data_blk += _l
            if not in_datablk:
                break
            if data_blk != '':
                cdata = yaml.load(data_blk, Loader=yaml.CSafeLoader)[0]
            else:
                continue
            t_stamp = cdata[0][0]
            cluster = cdata[1]
            if t_stamp <= t_stamp0:
                fdata += cluster  # append new cluster
            else:
                t_stamp0 = t_stamp
                yield fdata  # deliver completed frame
                fdata = cluster  # initialize next frame

    @staticmethod
    def Advacam_clog(file):
        """Read data in Advacam .clog format and yield frame

        Args:
        * file: file handle
        """

        width = 256
        frame = []
        while True:
            _l = file.readline()
            if not _l:
                break
            if isinstance(_l, bytes):
                _l = _l.decode()  # needed for gzip returning bytes objects
            if _l == '':  # skip empty lines between frames
                pass
            elif _l[0:5] == "Frame":  # new start-of-frame found
                if frame != []:
                    yield frame
                    frame = []
            elif _l[0] == '[':  # new cluster
                _l = re.sub(r"[^0-9, -, \[, \], \.]", '', _l.replace(', ', ','))  # only leave valid chars before eval()
                for _p in _l.split():
                    _pxl = eval(_p)
                    frame.append([int(_pxl[0] + _pxl[1] * width), int(_pxl[2])])
        file.close()

    # function to read frame data in advacam .txt (sparse matrix) format
    @staticmethod
    def Advacam_txt(file):
        """Read data in Advacam .txt (sparse matrix) format and pixel frames

        A frame contains lines with pairs of pixel number and pixel value;
        frames are separated by a line containing a '#'

        Args:
        * file: file handle
        """

        frame = []
        while True:
            _l = file.readline()
            if not _l:
                break
            if isinstance(_l, bytes):
                _l = _l.decode()  # needed for gzip returning bytes objects
            if _l != '#\n':  # not end of frame
                # add pixel number and value to current pixel list
                frame.append([int(_l.split('\t')[0]), int(_l.split('\t')[1])])
            else:
                yield frame
                frame = []
        file.close()


def pxl2map(pxlist):
    """Create pixel energy map from pixel list

    Args:
      - pxlist: list of pixels [ ..., [px_idx, px_energy], ...]
      - num: int, for numbering figures

    Returns:
      - numpy array
    """

    # get coordinates of pixels from pixel indices
    xy_l = np.array([[_l[0] % 256, _l[0] // 256] for _l in pxlist])
    # remove offset
    xy_l[:, 0] -= min(xy_l[:, 0])
    xy_l[:, 1] -= min(xy_l[:, 1])
    # print(xylst)

    # get energies of pixels
    E_l = [_l[1] for _l in pxlist]

    # dimension of rectangle containing cluster
    nx = max(xy_l[:, 0]) + 1
    ny = max(xy_l[:, 1]) + 1

    # plot pixel map
    _cmap = np.zeros((ny, nx))
    for _i, _xy in enumerate(xy_l):
        _cmap[_xy[1], _xy[0]] = E_l[_i]
    return _cmap


def plot_cluster(pxlist, num=0):
    """Create pixel energy map from pixel list and (optionally) plot it

    Args:
      - pxlist: list of pixels [ ..., [px_idx, px_energy], ...]
      - num: int, for numbering figures

    Returns:
      - matplotlib figure
    """

    # convert pixel list to ndarray
    _cmap = pxl2map(pxlist)
    ny, nx = np.shape(_cmap)

    _fig, _axim = plt.subplots(1, 1, num=f"pxl_image{num}", figsize=(2.0 + nx * 1.0, 0.5 + ny * 1.0))
    _axim.set_xlabel("# x  ", loc="right")
    _axim.set_ylabel("# y  ", loc="top")
    vmin, vmax = 0.5, 500
    _img = _axim.imshow(_cmap, origin="lower", cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax), extent=[0, nx, 0, ny])
    _cbar = _fig.colorbar(_img, pad=0.05)
    _img.set_clim(vmin=vmin, vmax=vmax)
    # force integer tick marks
    _axim.set_xticks(range(0, nx + 1))
    _axim.set_yticks(range(0, ny + 1))
    return _fig


class clusterReader:
    """Read cluster data written with mPIXdaq, print meta data and statistics

    Default cuts on cluster features are used to classify clusters as ɑ, β or γ signatures.

      ɑ: round cluster shape and peaking energy distribution, high ionization per pixel
      β: shape is not round, low ionization per pixel, >5 pixels
      γ: not (ɑ or β)

    Methods:

       * __init__(): instantiate class clusterReader and - optionally - set input file name
       * set_cuts(): set cut values
          - small_cut: separate small and large clusters
          - circularity_cut: round topology
          - flatness_cut: flat energy distribution
          - emean_cut: energy loss per pixel (only used if emx not in feature list)
          - emx_cut: maximum pixel energy
          - no_saturation:  ignore clusters with saturated pixels
       * parse_args():  read command line arguments if used interactively
       * read_data(): load data in yaml formt in pandas data frame
       * set_selection_masks(): define boolean masks to select ɑ, β and γ
       * get_statistics(): count signatures and provide parameters of energy distributions
       * plot(): plot energy distributions of ɑ, β and γ
       * __call__(): execute read_data(), set_selection_masks() and get_statistics()
    """

    def __init__(self, filename=None):
        self.filename = filename
        self.set_cuts()  # set default cut values

    def set_cuts(
        self, small_cut=4, circularity_cut=0.5, flatness_cut=0.5, emean_cut=200, emx_cut=400, no_saturation=False
    ):
        """Set default cut values"""

        self.small_cut = small_cut  # small clusters
        self.circularity_cut = circularity_cut  #  round topology
        self.flatness_cut = flatness_cut  # flat energy distribution
        self.emean_cut = emean_cut  #  cut on high energy loss per pixel
        self.emx_cut = emx_cut  # cut on maximum pixel energy
        self.no_saturation = no_saturation  #  ignore clusters with saturated pixels

    def parse_args(self):
        """parse command line arguments"""

        parser = argparse.ArgumentParser(description="read cluster data and show summary")
        parser.add_argument("filename")
        parser.add_argument('-v', '--verbosity', type=int, default=1, help='verbosity level (1)')
        parser.add_argument(
            '-c', '--circularity_cut', type=float, default=0.5, help='cut on circularity for alpha detection (0.5)'
        )
        parser.add_argument(
            '-s', '--small_cut', type=float, default=4, help='max. number of pixels for small cluster (4)'
        )
        parser.add_argument(
            '-f', '--flatness_cut', type=float, default=0.5, help='cut on flatness for alpha detection (0.6)'
        )
        parser.add_argument('--emean_cut', type=float, default=200, help='cut on mean pixel energy (keV)')
        parser.add_argument('-m', '--emx_cut', type=float, default=400, help='cut on maximum pixel energy (keV)')
        parser.add_argument(
            '-n', '--no-saturation', dest='no_saturation', action='store_true', help='neglect ɑ with saturated pixels'
        )
        args = parser.parse_args()
        self.filename = args.filename

        # *==* selection cuts ...
        self.small_cut = args.small_cut  # small clusters
        self.circularity_cut = args.circularity_cut  #  round topology
        self.flatness_cut = args.flatness_cut  # flat energy distribution
        self.emean_cut = args.emean_cut  #  cut on high energy loss per pixel
        self.emx_cut = args.emx_cut  # cut on maximum pixel energy
        self.no_saturation = args.no_saturation

    def read_data(self, fn=None, pr=True):
        """load yaml from file and fill pandas data frame

        Returns:
           -  data frame with cluster data
           -  meta-data dictionary
        """

        if fn is not None:
            self.filename = fn
        if self.filename is None:
            print("!!! no filename given - exit !")
            sys.exit("no filename given")
        else:
            fn = self.filename

        if '.csv' in self.filename:  # from csv file
            self.df = pd.read_csv(fn)
            self.meta_data = {}
            self.input_dict = None
        elif '.yml' in fn:  # or from file in yaml format
            f = gzip.open(fn, 'rb') if fn.split('.')[-1] == 'gz' else open(fn, 'r')
            self.input_dict = yaml.load(f, Loader=yaml.CLoader)
            if 'cluster_data' not in self.input_dict.keys():
                print("!!! no cluster data found in file - exiting!")
                sys.exit(1)
            self.keys = self.input_dict['keys']
            self.meta_data = self.input_dict["meta_data"]
            #  get cluster data: 1st is list of cluster properties, 2nd is list of [pixel index, energy] pairs
            cluster_data = self.input_dict['cluster_data']
            cluster_properties = [cluster_data[i][0] for i in range(len(cluster_data))]
            pixel_lists = [cluster_data[i][1] for i in range(len(cluster_data))]
            self.df = pd.DataFrame(data=cluster_properties, columns=self.keys)
            self.df['pixels'] = pixel_lists
        else:
            print("!!! unknown file format")
            self.df = None

        # *==* determine number of frames in data set
        self.n_frames = len(set(self.df['time']))  # use time stamps to count number of frames
        if self.input_dict is not None:
            self.acq_time = self.meta_data['acq_time']  # exposure time per frame
            self.T_alive = self.n_frames * self.acq_time
        else:
            self.T_alive = self.df['time'].to_numpy()[-1]

        # *==* remove empty records (i.e. frames without any cluster)
        is_not_nan = self.df['x_mean'].notna()
        n_empty = (~is_not_nan).sum()
        if n_empty > 0:
            print(f"removing empty records: {n_empty} removed")
            self.df = self.df[is_not_nan]

        # *==* determine total number of clusters in data set
        self.n_clusters = len(self.df)
        # *==* add some (useful) derived features to quantify cluster properties
        #  - mean pixel energy
        self.df['Epix_mean'] = self.df['energy'] / self.df['n_pix']
        #  - circularity defined as the ratio of the smaller and the larger eigenvalue
        #    of the covariance matrix of the pixels in a cluster
        self.df['circularity'] = self.df['var_mn'] / np.maximum(self.df['var_mx'].to_numpy(), 0.001)
        #  - flatness (of energy distribution) as the ratio of maximum variances
        #    of pixel and energy distributions in clusters
        self.df['flatness'] = self.df['varE_mx'] / np.maximum(self.df['var_mx'].to_numpy(), 0.001)
        #  - straightness"
        if 'w' in self.df.keys():
            self.df['straightness'] = self.df[['w', 'h']].max(axis=1) / self.df['n_pix']

        # *==* meta data
        self.meta_data['N_frames'] = self.n_frames
        self.meta_data['N_clusters'] = self.n_clusters
        self.meta_data['T_alive'] = self.T_alive

        if pr:
            print("\n*==* Contents of file", fn)
            if self.input_dict is not None:
                s_time = f"{self.meta_data['time']}  "
                s_frames = f"{self.n_frames} frames of {self.meta_data['acq_time']}s exposure time"
                s_device = f"{self.input_dict['deviceInfo']['dn']}"
                print("  Data written on " + s_time + "with device  " + s_device)
                print("  " + s_frames)
                print(f"  {self.n_clusters} clusters -> rate = {self.n_clusters / self.T_alive:.3g} Hz")
                print("\n*==* cluster features:\n  ", self.df.keys())
                print()
            else:
                print("   no device information available")
                print(f"  {self.n_frames} frames recorded in {self.meta_data['T_alive']}s")
                print(f"  {self.n_clusters} clusters -> rate = {self.n_clusters / self.T_alive:.3g} Hz")
                print("\n*==* cluster features:\n  ", self.df.keys())

        return self.df, self.meta_data

    def set_selection_masks(self):
        """classification of cluster types"""

        # *==* set boolean masks for selection
        is_small = self.df['n_pix'] <= self.small_cut  #
        is_circular = self.df['circularity'] >= self.circularity_cut  # circular shape
        is_flat = self.df['flatness'] > self.flatness_cut  # flat energy distribution
        if 'e_mx' in self.df.keys():
            is_high_dEdx = self.df['e_mx'] > self.emx_cut  # high energy loss per pixel
        else:
            is_high_dEdx = self.df['Epix_mean'] > self.emean_cut  # high energy loss per pixel

        # *==* definition of ɑ candidates
        # alphas have
        #   - a round cluster shape with
        #   - a peaking energy distribution and
        #   - a large value of the maximum pixel energy
        shape_is_alpha = is_circular & ~is_flat
        # a loose ɑ definition based only on dEdx
        is_cand_alpha = is_high_dEdx
        # a tight ɑ definition as the logical 'and' of criterea
        is_alpha = shape_is_alpha & is_high_dEdx
        # avoid non-linearity of response if max. pixel energy is too high
        if 'e_mx' in self.df.keys():
            is_saturating = self.df['e_mx'] > 1200
        else:
            is_saturating = self.df['Epix_mean'] > 210
        is_clean_alpha = is_alpha & ~is_saturating

        # *==* definition of β candidates (long non-alpha tracks with low small dEdx)
        shape_is_beta = ~shape_is_alpha & ~is_small
        is_beta = shape_is_beta & ~is_high_dEdx  #  request β shape  & low energy deposits

        # *==* define γ candidates (low-multiplicity clusters with small dEdx)
        is_gamma = is_small & ~is_high_dEdx

        # *==* define muon candidates (ling, straight beta-like traces)
        if 'straightness' in self.df.keys():
            is_muon = is_beta & (self.df['straightness'] > 0.9) & (self.df['n_pix'] > 25)
        else:
            is_muon = None

        # export selection
        if self.no_saturation:
            self.sel_alpha = is_clean_alpha  # well-measured alphas only
        else:
            self.sel_alpha = is_alpha
        # self.sel_alpha = is_cand_alpha  # alternative: selection based only on dEdx
        self.sel_beta = is_beta
        self.sel_gamma = is_gamma
        self.sel_muon = is_muon

    def __call__(self):
        """read read data, print meta data, initialize selection cuts, collect and print statistics"""
        # read data
        self.read_data(self.filename)
        # set selection criterea
        self.set_selection_masks()
        # collect and print
        self.get_statistics()

    def get_statistics(self, pr=True):
        """Collect statistics, construct result dictionary and print

        Returns:
           result dictionary: events, rate, mean and sigma of energy distribution for ɑ, β, γ
        """

        # result dictionary
        d = {'alpha': {}, 'beta': {}, 'gamma': {}}

        # create 2.5 sigma truncated samples (to avoid outliers)
        _k = 'energy'
        self.c_alpha, _low, _high = sigmaclip(self.df[self.sel_alpha][_k], 2.5, 2.5)
        self.c_beta, _low, _high = sigmaclip(self.df[self.sel_beta][_k], 2.5, 2.5)
        self.c_gamma, _low, _high = sigmaclip(self.df[self.sel_gamma][_k], 2.5, 2.5)

        # *==* collect statistics
        #  - number of events per class (number of True values in masks)
        self.N_alpha = int(self.sel_alpha.sum())
        self.N_beta = int(self.sel_beta.sum())
        self.N_gamma = int(self.sel_gamma.sum())
        d['alpha']['N'] = self.N_alpha
        d['beta']['N'] = self.N_beta
        d['gamma']['N'] = self.N_gamma
        #  - rates
        _tl = self.T_alive
        d['alpha']['r'] = self.N_alpha / _tl
        d['beta']['r'] = self.N_beta / _tl
        d['gamma']['r'] = self.N_gamma / _tl
        #  - trimmed mean energy
        d['alpha']['E'] = self.c_alpha.mean()
        d['beta']['E'] = self.c_beta.mean()
        d['gamma']['E'] = self.c_gamma.mean()
        #  - sigma of energy distribution
        d['alpha']['sE'] = self.c_alpha.std()
        d['beta']['sE'] = self.c_beta.std()
        d['gamma']['sE'] = self.c_gamma.std()

        if self.sel_muon is not None:
            self.N_muon = int(self.sel_muon.sum())
            d['muon'] = {}
            d['muon']['N'] = self.N_muon

        if not pr:
            return d

        # print cut values
        print("*==* selection cuts")
        print(f"  ɑ: flatness < {self.flatness_cut}")
        print(f"     circularity > {self.circularity_cut}")
        if 'e_mx' in self.df.keys():
            print(f"     emx_cut > {self.emx_cut} keV")
        else:
            print(f"     emean_cut > {self.emean_cut} keV")
        print(f"  β: size > {self.small_cut:.0f} and not ɑ")
        print("  γ: not(ɑ or β)")

        # print in tabular form
        print("\n*==* ɑ, β, γ Statistics:")
        print("             " + f"\t {'ɑ ':>10s} \t {'β ':>10s} \t {'γ ':>10s}")
        print("  events     " + f"\t {d['alpha']['N']:10d} \t {d['beta']['N']:10d} \t {d['gamma']['N']:10d}")
        print("  rate (Hz)  " + f"\t {d['alpha']['r']:10.3g} \t {d['beta']['r']:10.3g} \t {d['gamma']['r']:10.3g}")
        print("  meanE (keV)" + f"\t {d['alpha']['E']:10.3g} \t {d['beta']['E']:10.3g} \t {d['gamma']['E']:10.3g}")
        print("  sigE (keV)" + f"\t {d['alpha']['sE']:10.3g} \t {d['beta']['sE']:10.3g} \t {d['gamma']['sE']:10.3g}")
        if self.sel_muon is not None and self.N_muon > 0:
            print("  muons " + 8 * '\t' + f" {self.N_muon} µ")

        print(flush=True)
        return d

    def plot(self):
        """Graphical output"""

        def stacked_hists(key, _bins):
            """helper to produce stacked histograms for alpha, beta and gamma candidates"""
            _vals = (self.df[self.sel_alpha][key], self.df[self.sel_beta][key], self.df[self.sel_gamma][key])
            _labels = ('ɑ', 'β', 'γ')
            _colors = ('r', 'b', 'y')
            return plt.hist(_vals, label=_labels, bins=_bins, color=_colors, alpha=0.75, rwidth=0.75, stacked=True)

        # *==* histogram energy distributions
        _key = "energy"
        mx = min(max(self.df[self.sel_alpha][_key]), 4990)
        f = 10 ** int(np.log10(mx / 5))
        _bins = np.linspace(0, mx, int(mx / f) + 1)
        _rc_hist = stacked_hists(_key, _bins)
        plt.xlabel("Cluster energy (keV)")
        plt.ylabel(f"Counts / {f} keV")
        _tl = self.T_alive

        rate_txt = f"ɑ: {self.N_alpha / _tl:.2g} Hz, β: {self.N_beta / _tl:.2g} Hz, γ: {self.N_gamma / _tl:.2g} Hz"
        plt.text(0.25, 0.75, "rates " + rate_txt, transform=plt.gca().transAxes)
        energy_txt = (
            f"ɑ: {self.c_alpha.mean():.3g} keV, β: {self.c_beta.mean():.3g} keV, γ: {self.c_gamma.mean():.3g} keV"
        )
        plt.text(0.25, 0.70, "energies " + energy_txt, transform=plt.gca().transAxes)
        plt.title("Energy")
        plt.yscale('log')
        plt.legend()
        return plt.gcf()


class shmManager:
    """simple management of shared memory blocks

    class methods:
      - def get_sharedMem(name, size): create or link to shared memory

      do not forget to close() and finally unlink() all requested blocks
      in calling process
    """

    # list with names of all created memory blocks
    shm_names = []
    shms = []

    @classmethod
    def get_sharedMem(cls, name, size=None):
        """Create if necessary and return link to buffer

        Args:
          - name of shared data block
          - size: size in bytes, not needed if shared memory already created

        Returns: shared memory object

        """
        if name not in cls.shm_names:  # create new shared memory block
            try:
                _shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            except FileExistsError:
                print(f"!!! warning: shared memory '{name}' already existed")
                _shm = shared_memory.SharedMemory(name=name, create=False, size=size)
                if _shm.size != size:  # wrong size, delete and re-create
                    _shm.close()
                    _shm.unlink()
                    _shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            except Exception as e:
                print("!!! failed to create shared memory, Error: " + str(e))
            cls.shms.append(_shm)
            cls.shm_names.append(cls.shms[-1].name)
            return _shm
        else:  # link to existing shared memory block
            _shm = shared_memory.SharedMemory(name=name)
        return _shm

    @classmethod
    def close_sharedMem(cls, name):
        """close shared memory block by name"""
        for _i in range(len(cls.shms)):
            if name == cls.shm_names[_i]:
                cls.shms[_i].close()
                break

    @classmethod
    def unlink_sharedMem(cls, name):
        """unlink shared memory block by name and remove from lists"""
        for _i in range(len(cls.shms)):
            if name == cls.shm_names[_i]:
                cls.shms[_i].unlink()
                del cls.shms[_i]
                del cls.shm_names[_i]
                break
