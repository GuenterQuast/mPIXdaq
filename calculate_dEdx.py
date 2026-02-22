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
    _fig = plt.figure("dEdx_electron", figsize=(6.0, 3.5))
    ax_dEdx_e = _fig.add_subplot()
    plt.suptitle("Energy loss of electrons (mod. Bethe)")
    fig_dEdx_e = pt.plot_dEdx_electron((mp.H2O, mp.Si, mp.Pb), axis=ax_dEdx_e)

    _fig = plt.figure("dEdx_alpha", figsize=(6.0, 3.5))
    ax_dEdx_alpha = _fig.add_subplot()
    plt.suptitle(rf"$\alpha$ energy loss & energy vs. penetration depth in air")
    fig_dEdx_alpha = pt.plot_dEdx_alpha(mp.air, axis=ax_dEdx_alpha)

    _fig = plt.figure("dE_pixels", figsize=(6.0, 3.5))
    ax_dE_pixels = _fig.add_subplot()
    E0 = 1.0
    plt.suptitle(f"Energy deposit Si-pixels for {E0:.2f} MeV β tracks", size="large")
    fig_dE_pixels = pt.plot_beta_pixel_energies(E0, axis=ax_dE_pixels)

    _fig = plt.figure("alpha_range_air", figsize=(6.0, 3.5))
    ax_alpha_range = _fig.add_subplot()
    plt.suptitle("Energy loss of α in air (Bethe-Bloch)")
    fig_alpha_range_air = pt.plot_alpha_range(mp.air, axis=ax_alpha_range)

    #  *** some control printout (just to compare numbers)
    verbose = 1
    if verbose:
        E0_e = 1.0
        print(f"Energy loss of electrons of {E0_e} MeV in water: ", end='')
        print(f"dE/dx = {mp.H2O['rho'] * pt.dEdx(E0_e, mp.H2O, mp.electron):.2f} MeV/cm")
        print(f"                                       in Si: ", end='')
        _dEdx_e = mp.Si['rho'] * pt.dEdx(E0_e, mp.Si, mp.electron)
        print(f"dE/dx = {_dEdx_e:.2f} MeV/cm   {_dEdx_e / mp.Si['w_eh'] / 10000:.0f} e-h pairs/µm")
        E0_a = 4.0
        _dEdx_a = mp.air['rho'] * pt.dEdx(E0_a, mp.air, mp.alpha)
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
