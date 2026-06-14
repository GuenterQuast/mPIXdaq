#!/usr/bin/env python
"""Read cluster data written with mPIXdaq, print meta data and statistics

Default cuts on cluster features are used to classify clusters as  ɑ, β or γ signatures.

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

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
import pandas as pd
import yaml
import gzip
import sys


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
            '--circularity_cut', type=float, default=0.5, help='cut on circularity for alpha detection (0.5)'
        )
        parser.add_argument('--small_cut', type=float, default=4, help='max. number of pixels for small cluster (4)')
        parser.add_argument('--flatness_cut', type=float, default=0.5, help='cut on flatness for alpha detection (0.6)')
        parser.add_argument('--emean_cut', type=float, default=200, help='cut on mean pixel energy (keV)')
        parser.add_argument('--emx_cut', type=float, default=400, help='cut on maximum pixel energy (keV)')
        parser.add_argument(
            '--no-saturation', dest='no_saturation', action='store_true', help='neglect ɑ with saturated pixels'
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

    def read_data(self, fn):
        """load yaml from file and fill pandas data frame"""

        if '.csv' in fn:  # from csv file
            self.df = pd.read_csv(fn)
            self.meta_data = None
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
        if self.meta_data is not None:
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

        # *==* meta data
        print("\n*==* Contents of file", fn)
        if self.meta_data is not None:
            s_time = f"{self.meta_data['time']}  "
            s_frames = f"{self.input_dict['eor_data']['Nframes']} frames of {self.meta_data['acq_time']}s exposure time"
            s_device = f"{self.input_dict['deviceInfo']['dn']}"
            print("  Data written on " + s_time + "with device  " + s_device)
            print("  " + s_frames)
            print(f"  {self.n_clusters} clusters -> rate = {self.n_clusters / self.T_alive:.3g} Hz")
            print("\n*==* cluster features:\n  ", self.keys)
            print()
        else:
            print("   no meta-data available")

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

        # export selection
        if self.no_saturation:
            self.sel_alpha = is_clean_alpha  # well-measured alphas only
        else:
            self.sel_alpha = is_alpha
        # self.sel_alpha = is_cand_alpha  # alternative: selection based only on dEdx
        self.sel_beta = is_beta
        self.sel_gamma = is_gamma

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
        print(f"  β: size > {self.small_cut} and not ɑ")
        print("  γ: not(ɑ or β)")

        # print in tabular form
        print("\n*==* ɑ, β, γ Statistics:")
        print("             " + f"\t {'ɑ ':>10s} \t {'β ':>10s} \t {'γ ':>10s}")
        print("  events     " + f"\t {d['alpha']['N']:10d} \t {d['beta']['N']:10d} \t {d['gamma']['N']:10d}")
        print("  rate (Hz)  " + f"\t {d['alpha']['r']:10.3g} \t {d['beta']['r']:10.3g} \t {d['gamma']['r']:10.3g}")
        print("  meanE (keV)" + f"\t {d['alpha']['E']:10.3g} \t {d['beta']['E']:10.3g} \t {d['gamma']['E']:10.3g}")
        print("  sigE (keV)" + f"\t {d['alpha']['sE']:10.3g} \t {d['beta']['sE']:10.3g} \t {d['gamma']['sE']:10.3g}")
        print()
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


if __name__ == "__main__":  # ------------------------------------------------------------------------
    # instantiate cluster file reader
    reader = clusterReader()
    # parse command line arguments
    reader.parse_args()
    print(" .... importing yaml, be patient ...")
    # read data, set selection masks, collect and print statistics
    reader()
    # plot energy distributions
    fig = reader.plot()
    plt.show()
