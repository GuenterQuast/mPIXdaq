#!/usr/bin/env python
#
# script to calculate the approximate energy loss in materials (dE/dx)
#  for heavy particles and electrons at low energies < ~5 MeV

# the most authoritative and comprehensive source today is:
#     https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html

import numpy as np
import matplotlib.pyplot as plt

import mpixdaq.physics_tools as pt

if __name__ == "__main__":  # -------------------------------------------------
    mp = pt.materials

    # *** produce graphs
    pt.plot_dEdx_electron((mp.H2O, mp.Si, mp.Pb))
    pt.plot_dEdx_alpha(mp.air)
    pt.plot_beta_pixel_energies()
    pt.plot_alpha_range(mp.air)

    #  *** some control printout (just to compare numbers)
    verbose = 1
    if verbose:
        E0_e = 1.0
        print(f"Energy loss of electrons of {E0_e} MeV in water: ", end='')
        print(f"dE/dx = {mp.H2O['rho'] * pt.dEdx(E0_e, mp.H2O, 1, mp.m_e):.2f} MeV/cm")
        print(f"                                       in Si: ", end='')
        _dEdx_e = mp.Si['rho'] * pt.dEdx(E0_e, mp.Si, mp.z_e, mp.m_e)
        print(f"dE/dx = {_dEdx_e:.2f} MeV/cm   {_dEdx_e / mp.Si['w_eh'] / 10000:.0f} e-h pairs/µm")
        E0_a = 4.0
        _dEdx_a = mp.air['rho'] * pt.dEdx(E0_a, mp.air, mp.z_alpha, mp.m_alpha)
        print(f"                    alphas of {E0_a} MeV in air: {_dEdx_a:.2f} MeV/cm", end='')
        print()

        # print("pixel energies in keV along track with {E0:.2f} MeV initial energy:")
        # for _i in range(n_px):
        #    print(f"{_i+1}: {E_px[_i]:.1f} ", end='')
        # print(f"        total deposited energy {E_px.sum():.1f} keV")
        # ---

    # *** show plots and wait for user
    plt.tight_layout()
    plt.ion()
    plt.show()
    _ = input(25 * ' ' + "type <ret> to end -> ")
