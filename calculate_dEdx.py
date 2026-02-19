#!/usr/bin/env python
#
# script to calculate the approximate energy loss in materials (dE/dx)
#  for heavy particles and electrons at low energies < ~5 MeV

# the most authoritative and comprehensive source today is:
#     https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------


def effective_Z_over_A():
    """Effective Z/A for air (Bragg additivity)"""

    Z_total = 0.0
    A_total = 0.0
    for fraction, A in air_composition.values():
        if A == 28.0134:
            Z = 7 * 2
        elif A == 31.9988:
            Z = 8 * 2
        elif A == 39.948:
            Z = 18
        else:
            Z = 0
        Z_total += fraction * Z
        A_total += fraction * A
    return Z_total / A_total


def bethe_bloch(E_MeV, Z_over_A, I, z, m):
    """Bethe-Bloch relation for heavy particles (p, µ, 𝛼),
    normalized to material density,

    Parameters:
      E: kinetic energy in MeV
      Z_over_A: effective Z/A
      I: mean ionization energy (MeV)
      z: charge of projectile
      m: mass of projectile (alpha oder muon)
    """

    K = 0.307075  # MeV mol^-1 cm^2
    m_e = 0.511  # MeV (electron mass)
    gamma = 1 + E_MeV / m
    beta2 = 1 - 1 / gamma**2

    term1 = (K * Z_over_A * z**2) / beta2
    argument = (2 * m_e * beta2 * gamma**2) / I

    dEdx = term1 * (np.log(argument) - beta2)
    return dEdx  # MeV cm²/g


def dedx_electron(E_kin, Z, A, I_ev):
    """Calculates the specific energy loss of electrons (beta radiation)
    normalized to material density,

    Result validated against
        https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html

    Parameters:
      E_kin : float - Kinetic energy of electrons in MeV
      Z     : int   - atomic number of material
      A     : float - atomic mass of material in g/mol
      I_ev  : float - average ionization potential in eV
    """

    # Physics constants
    m_e_c2 = 0.511  # electron mass in MeV/c²
    K = 0.307075  # 4 * pi * N_A * r_e^2 * m_e * c^2 in MeV cm^2 / mol

    # relativistic variables
    gamma = (E_kin / m_e_c2) + 1
    beta = np.sqrt(1 - (1 / gamma**2))
    # conversion of I from eV to MeV
    I_mev = I_ev * 1e-6

    # modified Bethe-equation for electrons (simplified for range < 10 MeV)
    #   contains  term for the indistinguishability of the electrons
    _term1 = np.log((E_kin**2 * (gamma + 1)) / (2 * I_mev**2))
    _term2 = (1 / gamma**2) * (1 + (E_kin**2 / 8) - (2 * gamma - 1) * np.log(2))

    # energy loss normalized to density (MeV cm^2 / g)
    return (K / 2) * (Z / A) * (1 / beta**2) * (_term1 + _term2)


def calc_pixel_energies(E0):
    """calculete pixel energies for a track with energy E0"""
    n_px = 0
    E_px = []
    E_x = E0
    px_size = 0.0055  # in cm
    while E_x > 0.0:
        dE = rho_Si * dedx_electron(E_x, Z_Si, A_Si, I_Si) * px_size
        if dE > E_x:
            dE = E_x
        n_px += 1
        E_px += [1000.0 * dE]  # in keV
        E_x -= dE
    else:
        return np.asarray(E_px)


def calc_E_vs_depth(E0, rho, Z_over_A, I, z, m):
    """calculate alpha energies after penetrating depth x of air
    Parameters:
    E0: initial alpha energy
    rho: material density
    Z_over_A: effective Z/A
    I: mean ionization energy (MeV)
    z: charge of projectile
    m: mass of projectile (alpha oder muon)
    """

    dx = 0.05
    x = np.linspace(0, 100 * dx, 100)  # distances in cm
    Ex = []
    E_left = E0
    i = 0
    while E_left > 0.15:
        dE = rho * bethe_bloch(E_left, Z_over_A, I, z, m) * dx
        if dE > E_left:
            dE = E_left
        Ex += [E_left]
        E_left -= dE
    else:
        return np.asarray(Ex)


if __name__ == "__main__":  # -------------------------------------------------
    # Material parameters
    # --- Silicon
    Z_Si = 14  # 4.15 # effective atomic number (Z=14)
    A_Si = 28  # Atommasse
    rho_Si = 2.33  # g/cm³
    I_Si = 173  # Ionisation potential in eV
    w_eh = 3.6e-6  # MeV per e-h-Paar (3.6 eV)

    # --- Water/tissue ---
    Z_H2O = 7.4  # effective atomic number
    A_H2O = 12.5  # effective atomic masse
    rho_H2O = 1.0
    I_H2O = 75.0  # eV

    # --- air
    rho_air = 1.225e-3  # g/cm³, density of air normal pressure
    # Composition of air
    air_composition = {"N2": (0.755, 14.0067 * 2), "O2": (0.231, 15.9994 * 2), "Ar": (0.0128, 39.948)}
    Z_over_A_air = effective_Z_over_A()
    I_air = 85.7e-6  # MeV, mean ionization energy of air

    # projectile
    z_alpha = 2.0  # charge of alpha
    m_alpha = 3727.4  # mass of alpha in MeV/c^2 (4 u)

    z_mu = 1  # charge of muon
    m_mu = 105.658  # mass of muon in MeV/c^2

    # graphs

    # -- dE/dx * rho Grafik
    bw = 0.05  # steps of 50 keV
    nb = 100
    xp = np.linspace(bw, nb * bw, num=nb, endpoint=True) + bw / 2.0
    fig_dEdx = plt.figure()
    plt.plot(xp, dedx_electron(xp, Z_H2O, A_H2O, I_H2O), '-', label=r"H$_2$O")
    plt.plot(xp, dedx_electron(xp, Z_Si, A_Si, I_Si), '-', label="Si")
    plt.xlabel("E [MeV]")
    plt.ylabel(r"enery loss  dE/dx$\,$/$\rho$   [MeV$\,$cm²/g]")
    plt.suptitle("Energy loss of electrons (mod. Bethe)")
    plt.legend()

    # --- pixel energies
    E0 = 1.5  # in MeV
    E_px = calc_pixel_energies(E0)
    n_px = len(E_px)
    fig_px = plt.figure()
    plt.bar(range(n_px), E_px, color='darkred')
    plt.xlabel("pixel number")
    plt.ylabel(r"pixel energy [keV]")
    plt.suptitle(f"track energy {E0:.2f} MeV", size="large")

    # dE/dx for alpha particles in air
    bw = 0.25
    nb = 40
    xp = np.linspace(bw, nb * bw, num=nb, endpoint=True) + bw / 2.0
    plt.figure()
    plt.plot(xp, rho_air * bethe_bloch(xp, Z_over_A_air, I_air, z_alpha, m_alpha), '-', label=r"$\alpha$")
    plt.plot(xp, rho_air * bethe_bloch(xp, Z_over_A_air, I_air, z_mu, m_mu), '-', label="µ")
    plt.xlabel(" α energy(MeV)")
    plt.ylabel("dE/dx (MeV/cm)")
    plt.suptitle("Energy loss in air (Bethe-Bloch)")
    plt.legend()

    # alpha energy after penetration depth in air
    E0 = 5.0  # initial energy in MeV
    dx = 0.05
    Ex = calc_E_vs_depth(E0, rho_air, Z_over_A_air, I_air, z_alpha, m_alpha)
    plt.figure()
    xp = [dx * i for i in range(len(Ex))]
    plt.bar(xp, Ex, color="darkblue", width=dx * 0.75)
    plt.ylabel("α energy (MeV)")
    plt.xlabel("penetration depth (cm)")
    plt.suptitle(rf"$\alpha$ energy vs. penetration depth in air")

    # --- some control printout (just to compare numbers)
    E0 = 1.0
    print(f"Energy loss of electrons of {E0} MeV in water: ", end='')
    print(f"dE/dx = {rho_H2O * dedx_electron(E0, Z_H2O, A_H2O, I_H2O):.4f} MeV/cm")
    print(f"                                       in Si: ", end='')
    dEdx = rho_Si * dedx_electron(E0, Z_Si, A_Si, I_Si)
    print(f"dE/dx = {dEdx:.4f} MeV/cm   {dEdx / w_eh / 10000:.0f} e-h pairs/µm")
    print(f"                    alphas of {4} MeV in air: ", end='')

    # print("pixel energies in keV along track with {E0:.2f} MeV initial energy:")
    # for _i in range(n_px):
    #    print(f"{_i+1}: {E_px[_i]:.1f} ", end='')
    # print(f"        total deposited energy {E_px.sum():.1f} keV")
    # ---

    plt.grid(True)
    plt.tight_layout()
    plt.show()
