#!/usr/bin/env python
#
# script to calculate the approximate energy loss in materials (dE/dx)
#  for heavy particles and electrons at low energies < ~5 MeV

# the most authoritative and comprehensive source today is:
#     https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------


class material_properties:
    """Collect properties of target materials and projectiles"""

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
    # Z_over_A_air = material_properties.effective_Z_over_A()
    I_air = 85.7e-6  # MeV, mean ionization energy of air

    # projectile
    z_alpha = 2  # charge of alpha
    m_alpha = 3727.4  # mass of alpha in MeV/c^2 (4 u)

    z_mu = 1  # charge of muon
    m_mu = 105.658  # mass of muon in MeV/c^2

    def __init__(self):
        print("\n *==* calulate_dEdx: initializing materials \n")
        Z_over_A_air = material_properties.effective_Z_over_A()

    @classmethod
    def effective_Z_over_A(cls):
        """Effective Z/A for air (Bragg additivity)"""
        Z_total = 0.0
        A_total = 0.0
        for fraction, A in cls.air_composition.values():
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
        cls.Z_over_A_air = Z_total / A_total


def Bethe_Bloch(T, Z_over_A, I, z, m):
    """Bethe-Bloch relation for heavy particles (p, µ, 𝛼),
    normalized to material density,

    Parameters:
      T: kinetic energy in MeV
      Z_over_A: effective Z/A
      I: mean ionization energy (MeV)
      z: charge of projectile
      m: mass of projectile (alpha oder muon)
    """

    K = 0.307075  # MeV mol^-1 cm^2
    m_e = 0.51099895  # MeV (electron mass)
    gamma = 1 + T / m
    beta2 = 1 - 1 / gamma**2

    # Maximum kinetic energy transfer T_max
    #     Formula for a heavy projectile (m_alpha >> m_e)
    r_mass = m_e / m
    T_max = 2 * m_e * beta2 * gamma**2 / (1 + 2 * gamma * r_mass + r_mass**2)

    term1 = (K * Z_over_A * z**2) / beta2
    term_ln = 0.5 * np.log(2 * m_e * beta2 * gamma**2 * T_max / I**2)

    return term1 * (term_ln - beta2)  # MeV cm²/g


def dEdx_electron(T, Z, A, I_eV):
    """Calculates the specific energy loss of electrons (beta radiation)
    normalized to material density,

    Result validated against
        https://physics.nist.gov/PhysRefDaqta/Star/Text/ESTAR.html

    Notes:
      - only collisions considered, no radiation loss (relevant for energy >>1 MeV)
      - density effect correction (delta) omitted here for simplicity
      - pure Bethe loss decreases below 0.4 MeV and becomes even negative
        below 0.15 MeV for alpha particles in air;
        this can be fixed by adding Barkas corrections, which are, however, not implemented here

    Formula: Bethe-Bloch equation modified for electrons, ICRU Report 37 (1984).

    Parameters:
      T    : float or ndarray - Kinetic energy of the electron (MeV)
      Z    : int              - Atomic number of the target material
      A    : float            - Atomic mass of the target material (g/mol)
      I_eV : float            - Mean excitation energy (eV)

    Returns:
      dEdx_col : float         - Mass collision stopping power (MeV*cm^2/g)
    """

    # Fundamental constants
    m_e = 0.51099895  # Electron rest mass in MeV
    K = 0.15353  # Constant 2*pi*N_A*r_e^2*m_e*c^2 in MeV*cm^2/mol

    # Relativistic parameters
    tau = T / m_e  # Kinetic energy in units of rest mass
    gamma = tau + 1  # Lorentz factor
    beta2 = 1 - 1 / (gamma**2)  # Velocity squared (v/c)^2
    I_MeV = I_eV * 1e-6  # Convert mean excitation energy to MeV

    # F(tau) function for electrons (ICRU 37)
    # Accounts for the identity of the primary and secondary (knock-on) electrons
    F_tau = 1 - beta2 + ((tau**2 / 8) - (2 * tau + 1) * np.log(2)) / (gamma**2)

    # Logarithmic term of the Bethe formula
    # Includes the maximum energy transfer (T/2 for electrons due to indistinguishability)
    log_term = np.log(tau**2 * (tau + 2) / (2 * (I_MeV / m_e) ** 2))

    # mass collision stopping power
    return (K * Z / (A * beta2)) * (log_term + F_tau)  # MeV cm²/g


def calc_pixel_energies(E0):
    """calculate pixel energies for an electron track with energy E0 in silicon"""
    n_px = 0
    E_px = []
    E_x = E0
    px_size = 0.0055  # in cm
    while E_x > 0.0:
        dE = mp.rho_Si * dEdx_electron(E_x, mp.Z_Si, mp.A_Si, mp.I_Si) * px_size
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
    E_curr = E0  # current energy
    i = 0
    dE_last = 0.0
    while E_curr > 0.1:
        _dE = rho * Bethe_Bloch(E_curr, Z_over_A, I, z, m) * dx
        dE = _dE if _dE > dE_last else dE_last  # avoid problem of simple Bethe-Bloch at low Energies
        dE_last = dE
        if dE > E_curr:
            dE = E_curr
        Ex += [E_curr]
        E_curr -= dE
    return np.asarray(Ex)


if __name__ == "__main__":  # -------------------------------------------------
    # *** initialize material properties
    mp = material_properties()

    # *** produce graphs

    # -- dE/dx * rho Grafik
    bw = 0.05  # steps of 50 keV
    nb = 100
    xp = np.linspace(bw, nb * bw, num=nb, endpoint=True) + bw / 2.0
    fig_dEdx = plt.figure()
    plt.plot(xp, dEdx_electron(xp, mp.Z_H2O, mp.A_H2O, mp.I_H2O), '-', label=r"H$_2$O")
    plt.plot(xp, dEdx_electron(xp, mp.Z_Si, mp.A_Si, mp.I_Si), '-', label="Si")
    plt.grid(True)
    plt.legend()
    plt.suptitle("Energy loss of electrons (mod. Bethe)")
    plt.xlabel("E [MeV]")
    plt.ylabel(r"enery loss  dE/dx$\,$/$\rho$   [MeV$\,$cm²/g]")

    # --- pixel energies
    E0 = 1.5  # in MeV
    E_px = calc_pixel_energies(E0)
    n_px = len(E_px)
    fig_px = plt.figure()
    plt.bar(range(n_px), E_px, color='darkred')
    plt.grid(True)
    plt.suptitle(f"track energy {E0:.2f} MeV", size="large")
    plt.xlabel("pixel number")
    plt.ylabel(r"pixel energy [keV]")

    # dE/dx for alpha particles in air
    bw = 0.05
    nb = 100
    mn = 0.15
    xp = np.linspace(0.0, nb * bw, num=nb, endpoint=True) + mn
    plt.figure()
    plt.plot(xp, mp.rho_air * Bethe_Bloch(xp, mp.Z_over_A_air, mp.I_air, mp.z_alpha, mp.m_alpha), '-', label=r"$\alpha$")
    plt.plot(xp, mp.rho_air * Bethe_Bloch(xp, mp.Z_over_A_air, mp.I_air, mp.z_mu, mp.m_mu), '-', label="µ")
    plt.legend()
    plt.grid(True)
    plt.suptitle("Energy loss in air (Bethe-Bloch)")
    plt.xlabel(" α energy(MeV)")
    plt.ylabel("dE/dx (MeV/cm)")

    # alpha energy after penetration depth in air
    E0 = 5.0  # initial energy in MeV
    dx = 0.05
    Ex = calc_E_vs_depth(E0, mp.rho_air, mp.Z_over_A_air, mp.I_air, mp.z_alpha, mp.m_alpha)
    plt.figure()
    xp = [dx * i for i in range(len(Ex))]
    plt.bar(xp, Ex, color="darkblue", width=dx * 0.75)
    plt.ylabel("α energy (MeV)")
    plt.xlabel("penetration depth (cm)")
    plt.suptitle(rf"$\alpha$ energy vs. penetration depth in air")
    plt.grid(True)

    #  *** some control printout (just to compare numbers)
    verbose = 1
    if verbose:
        E0_e = 1.0
        print(f"Energy loss of electrons of {E0_e} MeV in water: ", end='')
        print(f"dE/dx = {mp.rho_H2O * dEdx_electron(E0_e, mp.Z_H2O, mp.A_H2O, mp.I_H2O):.2f} MeV/cm")
        print(f"                                       in Si: ", end='')
        _dEdx_e = mp.rho_Si * dEdx_electron(E0_e, mp.Z_Si, mp.A_Si, mp.I_Si)
        print(f"dE/dx = {_dEdx_e:.2f} MeV/cm   {_dEdx_e / mp.w_eh / 10000:.0f} e-h pairs/µm")
        E0_a = 4.0
        _dEdx_a = mp.rho_air * Bethe_Bloch(E0_a, mp.Z_over_A_air, mp.I_air, mp.z_alpha, mp.m_alpha)
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
