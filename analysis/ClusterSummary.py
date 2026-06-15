#!/usr/bin/env python
"""Read cluster data written with mPIXdaq, print meta data and statistics

Default cuts on cluster features are used to classify clusters as  ɑ, β or γ signatures.

  ɑ: round cluster shape and peaking energy distribution, high ionization per pixel
  β: shape is not round, low ionization per pixel, >5 pixels
  γ: not (ɑ or β)

Methods of class clusterReader() from package mpixdaq.mpixhelpers:

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

from mpixdaq.mpixhelpers import clusterReader
import matplotlib.pyplot as plt

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
