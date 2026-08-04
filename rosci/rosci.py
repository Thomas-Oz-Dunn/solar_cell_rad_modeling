# Radiation-Optimizing Solar Cell Investigator (ROSCI)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.constants as const

def wavelength_to_freq(wavelength):
    return const.c / wavelength

def freq_to_energy(frequency):
    return const.h * frequency

def wavelength_to_energy(wavelength):
    return freq_to_energy(wavelength_to_freq(wavelength))

def energy_to_wavelength(energy):
    freq = energy / const.h
    return const.c / freq

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
        return (1-x) * a_A + x * a_B - b * x *(1 - x)


def short_circuit_current(I_01, I_02, V_OC, ideality_factor, temperature):
    """
    Calculate short circuit current from multi diode model

    Parameters
    ----------
    I_01
        Base diode 1 current

    I_02
        Base diode 2 current

    V_OC
        Open circuit voltage

    ideality_factor
        Diode ideality factor

    temperature
        Temperature in kelvin

    TODO-TD: label source of equation
    TODO-TD: double check units, Kelvin?
    TODO-TD: vectorize to take in np.arrays instead of just scalar?
    """
    V_t = (const.k * temperature) / const.e

    # diffusion current in qausi-neutral region
    diffusion_current = I_01 * np.exp(V_OC / V_t) 

    # defect driven Recomb current  in depletion region
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

    Impact of device design parameters on quantum efficiency of solar cell and revelation of recombination mechanism

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

    TODO-TD: double check the units of h
    """
    photon_energy = wavelength_to_energy(wavelength)
    return thickness * (absorption_coeff - photon_energy  * recomb_rate / incident_power)

def calc_InP_absorbtion_coeff_300K(frequency):
    """
    Calculate alpha for InP at 300K

    from Yamaguchi,  Uemera 1984
    Electron Irradiation Damage in
    Radiation-Resistant InP Solar Cells

    """
    eV = freq_to_energy(frequency)
    if eV > 1.58:
        return 1.1e7 * np.exp(-9.9 / eV)
    elif eV > 1.31:
        return 4e4 * np.sqrt(eV - 1.31)
    else:
        return None

    
def eqe_dynamic_alpha_InP_300K(
    thickness, 
    wavelength, 
    recomb_rate, 
    incident_power
):
    """
    TODO-TD: vectorize across an array spectrum?
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
    
    """
    df = pd.read_csv('./wmo.csv', delim_whitespace=True)
    wavelengths = df['nm']
    power_density = df['W/sm/nm']
    return eqe_dynamic_alpha_InP_300K(
        thickness, 
        wavelengths, 
        recomb_rate, 
        power_density
    )

def plot_bandgap_lattice_constant(out_path = './Bandgaps_Lattice.png'):
    # TODO-TD: Use vegards_law to interpolate alloys
    # Vurgaftman 2001
    # Band parameters for III–V compound semiconductors and their alloys for curvature
    df = pd.read_csv('data\semiconductors.csv')
    df['Bandgap (Eg)'] = df['Bandgap (Eg)'].str.replace(' eV', '').astype(float)

    color_dict = {
        'Diamond (FCC)': 'red', 
        'Zinc blende (FCC)': 'blue', 
        'Wurtzite': 'green',
    }

    plt.figure(figsize=(8, 6))

    for structure, color in color_dict.items():
        subset = df[df['Crystal Structure'] == structure]
        plt.scatter(
            subset['a (A)'],
            subset['Bandgap (Eg)'],
            color=color,
            s=50,
            label=structure
        )


    for _, row in df.iterrows():
        plt.annotate(
            row['Material'], 
            (row['a (A)'], row['Bandgap (Eg)']),
            textcoords="offset points", 
            xytext=(0, 10)
        )

    plt.legend(title="Crystal Structure")
    plt.title('Semiconductor Bandgap vs. Lattice Constant (a)')
    plt.xlabel('Lattice Constant a (A)')
    plt.ylabel('Bandgap $E_g$ (eV)')
    plt.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)


def plot_bandgaps_on_spectrum(out_path = './Bandgaps_AMO0.png'):

    amo_df = pd.read_csv(Path('./data/wmo.csv'), delim_whitespace=True)
    material_df = pd.read_csv(Path('./data/semiconductors.csv'))

    material_df['Bandgap (Eg)'] = material_df['Bandgap (Eg)'].str.replace(' eV', '').astype(float)


    plt.plot(
        amo_df['nm'][:15*len(amo_df['nm'])//16], 
        amo_df['W/sm/nm'][:15*len(amo_df['W/sm/nm'])//16], 
        color='k', 
        label='Solar AM0 $(1367 W/m^2)$'
    )
    mini = np.min(amo_df['W/sm/nm'])
    maxa = np.max(amo_df['W/sm/nm'])
    
    for row in enumerate(material_df.iterrows()):
        plt.vlines(
            energy_to_wavelength(row['Bandgap (Eg)']), 
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
    plt.savefig(out_path)
