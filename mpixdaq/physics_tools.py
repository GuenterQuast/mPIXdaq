# helper functions to calculate the energy deposition of particles in matter

# the most authoritative and comprehensive source today is:
#     https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html

import numpy as np
import matplotlib.pyplot as plt


class materials:
    """Collect properties of target materials and projectiles"""

    # dictionaries of material parameters

    # *** target materials

    # --- Silicon
    Si = {}
    Si['Z'] = 14  # atomic number
    Si['A'] = 28  # atomic mass
    Si['rho'] = 2.33  # density (g/cm³)
    Si['I'] = 173.0  # ionization potential (eV)
    Si['w_eh'] = 3.6e-6  # MeV per e-h-Paar (3.6 eV)
    Si['name'] = r"Si"

    # --- Aluminum
    Al = {}
    Al['Z'] = 13  # atomic number
    Al['A'] = 27  # atomic mass
    Al['rho'] = 2.699  # density (g/cm³)
    Al['I'] = 166.0  # ionization potential (eV)
    Al['name'] = r"Al"

    # --- Lead
    Pb = {}
    Pb['Z'] = 82  # atomic number
    Pb['A'] = 207.2  # atomic mass
    Pb['rho'] = 11.35  # density (g/cm³)
    Pb['I'] = 823.0  # ionization potential (eV)
    Pb['name'] = r"Pb"

    # --- water/tissue ---
    H2O = {}
    H2O['Z'] = 7.25  # effective atomic number
    H2O['A'] = 14.53  # effective atomic masse
    H2O['rho'] = 1.0
    H2O['I'] = 75.0  # ionization potential (eV)
    H2O['name'] = r"H$_2$O"

    # --- air
    air = {}
    air['Z'] = 7.25
    air['A'] = 14.53
    air['rho'] = 1.225e-3  # g/cm³, density of air normal pressure
    air['I'] = 85.7  # mean ionization energy of air (eV)
    air['name'] = r"air"

    # *** projectiles

    # --- alpha (He nucleus)
    alpha = {}
    alpha['z'] = 2
    alpha['m'] = 3727.4  # mass of alpha in MeV/c² (4 u)

    # --- proton
    proton = {}
    proton['z'] = 1
    proton['m'] = 938.272

    # --- muon
    muon = {}
    muon['z'] = 1  # charge of muon
    muon['m'] = 105.658  # mass of muon in MeV/c²

    # --- electron
    electron = {}
    electron['z'] = 1
    electron['m'] = 0.510999  # electron mass in MeV/c²


def dEdx(T, material, projectile):
    """calculate energy loss in matter ( or "mass stopping power")
    for heavy projectiles and electrons

    Parameters:
      T: kinetic energy in MeV
      Material dictionary
          Z: effective Z
          A: effective A
          I: mean ionization energy (MeV)
      z: charge of projectile
      m: mass of projectile

    Returns:
      mass stopping power in units of MeV cm²/g
      (to be multiplied by material density to obtain energy loss per unit length )

    Formulae:
    - Bethe-Bloch relation for heavy projectiles (m_alpha, m_p or m_µ >> m_e)
      Notes:
      - only collisions considered, no radiation loss (relevant for energy >>1 MeV)
      - density effect correction (delta) omitted here for simplicity
      - pure Bethe loss decreases below 0.4 MeV and becomes even negative
        below 0.15 MeV for alpha particles in air; this can be fixed by adding
        Barkas corrections, which are, however, not implemented here

    -  modified Bethe-Bloch equation for electrons, ICRU Report 37 (1984).
    """

    Z_over_A = material['Z'] / material['A']
    I = material['I'] * 1e-6
    z = projectile['z']
    m = projectile['m']

    # constants
    m_e = 0.51099895  # MeV (electron mass)
    K = 0.307075  # 4 pi N_A r_e² m_e c²  (MeV*cm^2/mol)
    # relativistic parameters
    tau = T / m  # Kinetic energy in units of projectile mass
    gamma = tau + 1  # Lorentz factor
    gamma2 = gamma**2
    beta2 = 1 - 1 / gamma2  # Velocity squared (v/c)^2
    C = K * Z_over_A * z**2 / beta2

    if m > 0.55:  #  Bethe-Bloch relation for heavy projectiles (m_alpha, m_p or m_µ >> m_e)
        # Maximum kinetic energy transfer T_max
        r_mass = m_e / m
        T_max = 2 * m_e * beta2 * gamma2 / (1 + 2 * gamma * r_mass + r_mass**2)
        term_ln = 0.5 * np.log(2 * m_e * beta2 * gamma2 * T_max / I**2)
        return C * (term_ln - beta2)  # MeV cm²/g

    else:  # modified Bethe-Bloch for electrons, ICRU Report 37 (1984).
        # F(tau) accounts for identity of the primary and secondary (knock-on) electrons
        F_tau = 1 - beta2 + ((tau**2 / 8) - (2 * tau + 1) * np.log(2)) / (gamma2)
        # maximum energy transfer is T/2 for indistinguishable electrons
        T_max = 0.5
        term_ln = 0.5 * np.log(tau**2 * (tau + 2) * m_e**2 * T_max / I**2)
        return C * (term_ln + F_tau)  # MeV cm²/g


def calc_pixel_energies(E0, px_size=0.0055):
    """calculate pixel energies for an electron track with energy E0 in silicon

    Parameters:
       E0: initial energy
       px_size: pixel size (cm)
    """
    mp = materials
    E_px = []
    E_x = E0
    while E_x > 0.0:
        dE = mp.Si['rho'] * dEdx(E_x, mp.Si, mp.electron) * px_size
        if dE > E_x:
            dE = E_x
        E_px += [1000.0 * dE]  # in keV
        E_x -= dE
    else:
        return np.asarray(E_px)


def calc_E_vs_depth(E0, dx, material, projectile):
    """calculate particle energies after penetrating depth x of material
    Parameters:
    E0: initial alpha energy
    dx: step width (cm)
    Material dictionary
        Z: effective Z
        A: effective A
        I: mean ionization energy (MeV)
        rho: density
    z: charge of projectile
    m: mass of projectile (alpha oder muon)
    """

    Ex = [E0]
    E_curr = E0  # current energy
    dE_last = 0.0
    while E_curr > 0.0:
        _dE = material['rho'] * dEdx(E_curr, material, projectile) * dx
        dE = _dE if _dE > dE_last else dE_last  # avoid problem of simple Bethe-Bloch at low energies
        dE_last = dE
        if dE > E_curr:
            dE = E_curr
        E_curr -= dE
        Ex += [E_curr]  # add to list
    return np.asarray(Ex)


def plot_dEdx_electron(material, nbins=100, bw=0.05, axis=None):
    """plot dE/dx * rho for electrons as a function of energy
    nbins: number of bins
    bw: bin width in MeV
    axis   : figure axis; a new one is created if not given

    returns: matplotlib figure
    """
    mp = materials
    xp = np.linspace(bw, nbins * bw, num=nbins, endpoint=True) + bw / 2.0
    if axis is None:
        fig_dEdx_e = plt.figure("dEdx_electron", figsize=(7.0, 4.5))
        ax_dEdx_e = fig_dEdx_e.add_subplot()
        plt.suptitle("Energy loss of electrons (mod. Bethe)")
    else:
        ax_dEdx_e = axis
    for _mp in material:
        ax_dEdx_e.plot(xp, dEdx(xp, _mp, mp.electron), '-', label=_mp['name'])
    plt.grid(True)
    plt.legend()
    plt.xlabel("E [MeV]", size='large')
    plt.ylabel(r"enery loss  dE/dx$\,$/$\rho$   [MeV$\,$cm²/g]", size='large')

    return plt.gcf()


def plot_beta_pixel_energies(E0=1.5, px_size=0.0055, axis=None):
    """energy deposits per pixel
    E0: initial energy
    px_size: pixel size in cm
    axis: figure axis; a new one is created if not given

    returns: matplotlib figure
    """
    E_px = calc_pixel_energies(E0, px_size=px_size)
    n_px = len(E_px)
    if axis is None:
        fig_dE_pixels = plt.figure("dE_pixels", figsize=(7.0, 4.5))
        ax_de_pixels = fig_dE_pixels.add_subplot()
        plt.suptitle(f"Energy deposit Si-pixels for {E0:.2f} MeV β tracks", size='large')
    else:
        ax_de_pixels = axis
    ax_de_pixels.bar(range(n_px), E_px, color='darkred', alpha=0.5)
    plt.grid(True)
    plt.xlabel(f"pixel number ({px_size * 1e4:.0f} µm / pixel)", size='large')
    plt.ylabel(r"pixel energy [keV]", size='large')

    return plt.gcf()


def plot_dEdx_alpha(material, nbins=200, bw=0.025, axis=None):
    """dE/dx for alpha particles in air vs. alpha energy
    material: target material
    nbins  : number of bins
    bw     : bin width
    axis   : figure axis; a new one is created if not given

    returns: matplotlib figure
    """
    mp = materials
    mn = 0.15  # simple Formula not valid for smaller values
    xp = np.linspace(0.0, nbins * bw, num=nbins, endpoint=True) + mn
    if axis is None:
        fig_dEdx_alpha = plt.figure("dEdx_alpha", figsize=(7.0, 4.5))
        ax_dEdx_alpha = fig_dEdx_alpha.add_subplot()
        plt.suptitle("Energy loss in air (Bethe-Bloch)")
    else:
        ax_dEdx_alpha = axis
    ax_dEdx_alpha.plot(xp, material['rho'] * dEdx(xp, material, mp.alpha), '-', label=r"$\alpha$")
    ax_dEdx_alpha.plot(xp, material['rho'] * dEdx(xp, material, mp.muon), '-', label="µ")
    plt.legend()
    plt.grid(True)
    plt.xlabel(" α energy(MeV)", size='large')
    plt.ylabel("dE/dx (MeV/cm)", size='large')

    return plt.gcf()


def plot_alpha_range(material, E0=6.0, dx=0.050, axis=None):
    """alpha energy after penetration depth in air
    material : target
    E0       : initial energy in MeV
    dx       : step width
    axis     : figure axis; a new one is created if not given

    returns: matplotlib figure
    """

    mp = materials
    # calculate energy loss ber bin
    Ex = calc_E_vs_depth(E0, dx, material, mp.alpha)
    if axis is None:
        fig_alpha_range_air = plt.figure("alpha_range_air", figsize=(7.0, 4.5))
        ax1_alpha_range = fig_alpha_range_air.add_subplot()
        plt.suptitle(rf"$\alpha$ energy loss & energy vs. penetration depth in {material['name']}")
    else:
        ax1_alpha_range = axis
    xp = [dx * i for i in range(len(Ex))]
    # plot particle energy
    ax1_alpha_range.bar(xp, Ex, color="darkblue", align='edge', width=dx * 0.5, alpha=0.5)
    ax1_alpha_range.set_ylabel("α energy (MeV)", color="darkblue", size='large')
    ax1_alpha_range.set_xlabel("material depth (cm)", size='large')
    # plot deposited energy(bin)
    ax2_alpha_range = ax1_alpha_range.twinx()
    ax2_alpha_range.bar(xp[:-1], Ex[:-1] - Ex[1:], align='edge', color="darkred", width=dx * 0.85, alpha=0.5)
    ax2_alpha_range.set_ylabel("deposited energy (MeV)", color="darkred", size='large')

    return plt.gcf()


if __name__ == "__main__":  # -------------------------------------------------
    # application example

    mp = materials
    # *** produce graphs
    plot_dEdx_electron((mp.H2O, mp.Si, mp.Pb))
    plot_beta_pixel_energies()
    plot_dEdx_alpha(mp.air)
    plot_alpha_range(mp.air)

    #  *** some control printout (just to compare numbers)
    verbose = 1
    if verbose:
        E0_e = 1.0
        print(f"Energy loss of electrons of {E0_e} MeV in water: ", end='')
        print(f"dE/dx = {mp.H2O['rho'] * dEdx(E0_e, mp.H2O, mp.electron):.2f} MeV/cm")
        print(f"                                       in Si: ", end='')
        _dEdx_e = mp.Si['rho'] * dEdx(E0_e, mp.Si, mp.electron)
        print(f"dE/dx = {_dEdx_e:.2f} MeV/cm   {_dEdx_e / mp.Si['w_eh'] / 10000:.0f} e-h pairs/µm")
        E0_a = 4.0
        _dEdx_a = mp.air['rho'] * dEdx(E0_a, mp.air, mp.alpha)
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
