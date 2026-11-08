# Radiation-Optimizing Solar Cell Investigator (ROSCI)

import numpy as np
import scipy.constants as const

def V_OC(I_SC, I_0, ideality, temperature):
    return ideality * const.k * temperature / const.q * np.ln(I_SC / I_0 + 1)

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
    TODO-TD: GaAs equivalent?
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

    # Vectorized
    condlist = [eV > 1.58, (eV > 1.31) & (eV <= 1.58)]
    choicelist = [1.1e7 * np.exp(-9.9 / eV), 4e4 * np.sqrt(eV - 1.31)]

    # Vectorized piecewise selection
    return np.select(condlist, choicelist, default=np.nan)


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

def diffusion_length_fluence(fluence, L_0, K_L):
    """
    1/L_phi^2 = 1/L_0^2 + K_L * phi
    L_0 ^2 = L_phi^2 ( 1 + L_0^2 * K_L * phi)
    L_phi^2 = L_0 ^2 / ( 1 + L_0^2 * K_L * phi)
    L_phi = sqrt( L_0 ^2 / ( 1 + L_0^2 * K_L * phi))
    L_phi = L_0 * sqrt( 1 / ( 1 + L_0^2 * K_L * phi))
    """
    den = 1 + L_0 * L_0 * K_L * fluence
    return L_0 * np.sqrt(1 / den)
