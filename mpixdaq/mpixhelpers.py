"""Helper functions for mpixdaq

- class fileDecoders to decode  various input file formats: mPIXdaq .npy and .yml and Advacam .txt and .clog
- function plot_cluster() to plot energy map of pixel cluster
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from multiprocessing import shared_memory
import re
import yaml


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


def plot_cluster(pxlist, num=0):
    """Plot energy map of pixel cluster

    Args:
      - pxlist: list of pixels [ ..., [px_idx, px_energy], ...]
      - num: int, for numbering figures

    Returns:
      - matplotlib figure
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
    _cimage = np.zeros((ny, nx))
    for _i, _xy in enumerate(xy_l):
        _cimage[_xy[1], _xy[0]] = E_l[_i]
    # print(_cimage)

    _fig, _axim = plt.subplots(1, 1, num=f"pxl_image{num}", figsize=(2.0 + nx * 1.0, 0.5 + ny * 1.0))
    _axim.set_xlabel("# x  ", loc="right")
    _axim.set_ylabel("# y  ", loc="top")
    vmin, vmax = 0.5, 500
    _img = _axim.imshow(_cimage, origin="lower", cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax), extent=[0, nx, 0, ny])
    _cbar = _fig.colorbar(_img, pad=0.05)
    _img.set_clim(vmin=vmin, vmax=vmax)

    return _fig


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
            _shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            cls.shms.append(_shm)
            cls.shm_names.append(cls.shms[-1].name)
            return _shm
        else:  # link to existing shared memory block and return as properly shaped ndarray
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
