from pathlib import Path
import numpy as np
import pandas as pd

from rosci.utils import vegards_law, parse_alloy, thermal_voltage

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "semiconductors.csv"

class Semiconductor:
    def __init__(
        self, 
        base_material_name, 
        doping_material_name, 
        doping_percent
    ):
        # e.g. {"GaAs": 0.7, "AlAs": 0.3}
        self.base_material = parse_alloy(base_material_name)      
        self.doping_material = parse_alloy(doping_material_name)
        self.doping_percent = doping_percent

    def calc_bandgap(self, semiconductor_data=DEFAULT_DATA):
        df_mat = pd.read_csv(semiconductor_data).set_index("Material")

        if len(self.base_material) != 2:
            raise NotImplementedError(
                "Vegard's law is only implemented for binary mixtures "
                f"(got {len(self.base_material)} components)."
            )
            # TODO-TD: extend

        (comp1, _), (comp2, frac2) = self.base_material.items()

        missing = [c for c in (comp1, comp2) if c not in df_mat.index]
        if missing:
            raise KeyError(f"{missing} not found in database. Available: {list(df_mat.index)}")

        eg1 = df_mat.loc[comp1, "Bandgap (Eg)"]
        eg2 = df_mat.loc[comp2, "Bandgap (Eg)"]
        return vegards_law(eg1, eg2, frac2)


class PNJunction:
    def __init__(
        self, 
        n_type_material: Semiconductor, 
        p_type_material: Semiconductor, 
        n_type_dims, 
        p_type_dims
    ):
        self.n_type_material = n_type_material
        self.p_type_material = p_type_material
        self.n_type_dims = n_type_dims
        self.p_type_dims = p_type_dims

    @staticmethod
    def thermal_voltage(temperature=300.0):
        """V_T = kT/q (about 25.85 mV at 300 K)."""
        return thermal_voltage(temperature)

    def calc_diffusion_length(self, diffusivity, lifetime,):
        """L = sqrt(D * tau)"""
        return np.sqrt(np.asarray(diffusivity) * np.asarray(lifetime))

    def calc_current(
        self, 
        voltage, 
        saturation_current, 
        temperature=300.0, 
        ideality=1.0,
    ):
        """Shockley diode equation: I = I_s * (exp(V / (n * V_T)) - 1)."""
        v_t = self.thermal_voltage(temperature)
        return saturation_current * np.exp(np.asarray(voltage) / (ideality * v_t))