# Radiation-Optimizing Solar Cell Investigator (ROSCI)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.constants as const

def wavelength_to_freq(wavelength):
    return const.c / wavelength

def freq_to_energy(frequency):
    """
    Photon energy in Joules (E = h*f)
    """
    return const.h * frequency

def wavelength_to_energy(wavelength):
    """
    Photon energy in Joules
    """
    return freq_to_energy(wavelength_to_freq(wavelength))

def energy_to_wavelength(energy):
    """
    Parameters
    ----------
    energy
        Photon energy in Joules
        If you have eV, multiply by
        const.e first to convert to Joules.
    """
    freq = energy / const.h
    return const.c / freq

def joules_to_eV(energy_j):
    """
    Convert Joules -> electronvolts. 
    1 eV = const.e Joules
    """
    return energy_j / const.e

def eV_to_joules(energy_ev):
    """
    Convert electronvolts -> Joules.
    """
    return energy_ev * const.e


def vegards_law(
    a_A,
    a_B,
    x,
    b=None
):
    """
    Interpolate between materials
    Lattice constant calculator

    https://en.wikipedia.org/wiki/Vegard%27s_law
    a_(A_(1-x)B_x) = (1-x) a_A + x a_B

    Use b for bandgap bowing parameter
    """
    if b is None:
        return (1-x) * a_A + x * a_B
    else:
        return (1-x) * a_A + x * a_B - b * x * (1 - x)


def short_circuit_current(I_01, I_02, V_OC, ideality_factor, temperature,):
    """
    Calculate short circuit current from multi diode model

    Parameters
    ----------
    I_01
        Diode 1 (diffusion) reverse saturation current

    I_02
        Diode 2 (recombination) reverse saturation current

    V_OC
        Open circuit voltage

    ideality_factor
        Diode ideality factor (typically ~2 for diode 2, ~1 for diode 1)

    temperature
        Temperature in Kelvin. 
            const.k is Boltzmann's constant in J/K
            const.e is the elementary charge in C, 
            so k*T/e has units of volts, as required for V_t (the thermal voltage).
    """
    V_t = (const.k * temperature) / const.e

    # diffusion current in quasi-neutral region
    diffusion_current = I_01 * np.exp(V_OC / V_t) 

    # defect driven Recomb current in depletion region
    recombination_current = I_02 * np.exp(V_OC / (ideality_factor * V_t)) 
    
    return diffusion_current + recombination_current



def eqe(
    thickness,
    absorption_coeff,
    wavelength,
    recomb_rate,
    incident_power
):
    """
    Calculate External Quantum Efficiency (EQE) from Livingston 2025

    Impact of device design parameters on quantum efficiency of solar cell and 
    revelation of recombination mechanism

    Parameters
    ----------
    thickness
        thickness of material

    absorbtion_coeff
        material absorbtion coefficient alpha

    wavelength
        Wavelength of light

    recomb_rate
        Material recombination rate

    incident_power
        Incident power at wavelength
    """
    photon_energy = wavelength_to_energy(wavelength)
    return thickness * (absorption_coeff - photon_energy * recomb_rate / incident_power)

def calc_InP_absorbtion_coeff_300K(frequency):
    """
    Calculate alpha for InP at 300K

    from Yamaguchi, Uemera 1984
    Electron Irradiation Damage in
    Radiation-Resistant InP Solar Cells

    Parameters
    ----------
    frequency
        Photon frequency (Hz)

    Returns
    -------
    Absorption coefficient alpha, same shape as `frequency`. 
    Values below the 1.31 eV absorption edge return np.nan
    """
    frequency = np.asarray(frequency, dtype=float)

    eV = joules_to_eV(freq_to_energy(frequency))
    if eV > 1.58:
        return 1.1e7 * np.exp(-9.9 / eV)
    elif eV > 1.31:
        return 4e4 * np.sqrt(eV - 1.31)
    else:
        return np.nan


def eqe_dynamic_alpha_InP_300K(
    thickness,
    wavelength,
    recomb_rate,
    incident_power
):
    """
    EQE for InP at 300K using the wavelength-dependent alpha model above.
    """
    freq = wavelength_to_freq(wavelength)
    alpha = calc_InP_absorbtion_coeff_300K(freq)
    return eqe(
        thickness,
        alpha,
        wavelength,
        recomb_rate,
        incident_power
    )

def calc_InP_300K_eqe_dist(thickness, recomb_rate):
    """
    EQE across the full AM0 spectrum (data/wmo.csv) for InP at 300K.
    """
    df = pd.read_csv(Path('./data/wmo.csv'), delim_whitespace=True)
    wavelengths = df['nm'].to_numpy()
    power_density = df['W/sm/nm'].to_numpy()
    return eqe_dynamic_alpha_InP_300K(
        thickness,
        wavelengths,
        recomb_rate,
        power_density
    )

# TODO-TD: compare indirect and direct bandgaps?

def plot_bandgap_lattice_constant(
    crystal_types=None,
    alloys=None,
    semiconductor_data='../data/semiconductors.csv',
    ax=None,
):
    """
    PLot semiconductor bandgaps vs lattice constants
    
    Use vegards_law to interpolate alloys with 
    Vurgaftman 2001
    Band parameters for III-V compound semiconductors and their alloys
    """
    df = pd.read_csv(semiconductor_data)
    df['Bandgap (Eg)'] = df['Bandgap (Eg)'].str.replace(' eV', '').astype(float)

    color_dict = {
        'Diamond (FCC)': 'red',
        'Zinc blende (FCC)': 'blue',
        'Wurtzite': 'green',
    }
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for structure, color in color_dict.items():
        subset = df[df['Crystal Structure'] == structure]
        if crystal_types is not None:
            if structure not in crystal_types:
                continue

        ax.scatter(
            subset['a (A)'],
            subset['Bandgap (Eg)'],
            color=color,
            s=50,
            label=structure
        )

    for _, row in df.iterrows():
        if crystal_types is not None:
            if row['Crystal Structure'] not in crystal_types:
                continue

        ax.annotate(
            row['Material'],
            (row['a (A)'], row['Bandgap (Eg)']),
            textcoords="offset points",
            xytext=(0, 10)
        )

    if alloys is not None:
        # Expected format ("GaAs", "InAs", 0.5, bowing_parameter (Optional))
        # TODO-TD: Alloy dictionary / dataclass?

        df_mat = df.set_index('Material')
        bandgaps = []
        lattices = []

        for alloy in alloys:
            comp1 = alloy[0]
            comp2 = alloy[1]
            if comp1 not in df_mat.index:
                print(f"{comp1} not in database {df_mat.index}")
            if comp2 not in df_mat.index:
                print(f"{comp2} not in database {df_mat.index}")

            if comp1 in df_mat.index and comp2 in df_mat.index:
                eg1 = df_mat.loc[comp1, 'Bandgap (Eg)']
                eg2 = df_mat.loc[comp2, 'Bandgap (Eg)']
                a1 = df_mat.loc[comp1, 'a (A)']
                a2 = df_mat.loc[comp2, 'a (A)']

                x = alloy[2]

                if len(alloy) == 4:
                    alloy_bandgap = vegards_law(eg1, eg2, x, alloy[3])
                else:
                    alloy_bandgap = vegards_law(eg1, eg2, x)

                alloy_lattice = vegards_law(a1, a2, x)

                bandgaps.append(alloy_bandgap)
                lattices.append(alloy_lattice)

                # Triple braces?????
                # TODO-TD: create a formatting function to detect common bas parameters and merge
                alloy_name = f"$({comp1})_{{{1 - x:.2f}}}({comp2})_{{{x:.2f}}}$"

                ax.annotate(
                    alloy_name,
                    (alloy_lattice, alloy_bandgap),
                    textcoords="offset points",
                    xytext=(0, 10)
                )

        # TODO-TD: interp dotted line? 
        # bowing parameter & x sweep?
        ax.scatter(
            lattices,
            bandgaps,
            c='grey',
            s=45,
            label='Alloys'
        )

    ax.legend(title="Crystal Structure")
    plt.title('Semiconductor Bandgap vs. Lattice Constant (a)')
    plt.xlabel('Lattice Constant a (A)')
    plt.ylabel('Bandgap $E_g$ (eV)')
    plt.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    return ax


def plot_bandgaps_on_spectrum(
    semiconductor_data='./data/semiconductors.csv', 
    ax=None
):
    """
    Plot semiconductor bandgaps on solar spectrum
    """
    material_df = pd.read_csv(Path(semiconductor_data))
    amo_df = pd.read_csv(Path('./data/wmo.csv'), delim_whitespace=True)

    material_df['Bandgap (Eg)'] = material_df['Bandgap (Eg)'].str.replace(' eV', '').astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(
        amo_df['nm'][:15*len(amo_df['nm'])//16],
        amo_df['W/sm/nm'][:15*len(amo_df['W/sm/nm'])//16],
        color='k',
        label='Solar AM0 $(1367 W/m^2)$'
    )
    mini = np.min(amo_df['W/sm/nm'])
    maxa = np.max(amo_df['W/sm/nm'])

    for _, row in material_df.iterrows():
        eg_joules = eV_to_joules(row['Bandgap (Eg)'])
        wavelength_nm = energy_to_wavelength(eg_joules) * 1e9

        ax.vlines(
            wavelength_nm,
            mini,
            maxa,
            ls='--',
            label=fr"{row['Material']} $E_g$: {row['Bandgap (Eg)']} eV"
        )

    plt.xlabel('Wavelength (nm)')
    plt.title('Solar AM0 Spectral Irradiance\nWavelengths of semiconductor bandgap energies')
    plt.ylabel('Spectral Irradiance $(W/m^2/nm)$')
    plt.grid(ls='--', which='both')
    plt.legend()
    plt.xscale('linear')
    plt.tight_layout()
    return ax
