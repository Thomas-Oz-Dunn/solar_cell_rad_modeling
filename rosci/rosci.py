# Radiation-Optimizing Solar Cell Investigator
# TODO-TD: pyproject.toml

import numpy as np
import scipy.constants as const

def short_circuit_current(I_01, I_02, V_OC, ideality_factor, temperature):
    """
    Calculate short circuit current from multi diode model
    TODO-TD: double check units, Kelvin?
    """
    V_t = (const.k * temperature) / const.e

    # in qausi-neutral region
    diffusion_current = I_01 * np.exp(V_OC / V_t) 
    # defect drive in depletion region
    recombination_current = I_02 * np.exp(V_OC / (ideality_factor * V_t)) 
    
    return diffusion_current + recombination_current

