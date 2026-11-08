import pandas as pd
from rosci.utils import vegards_law, parse_alloy



class Semiconductor:
    def __init__(
        self,
        base_material_name,
        doping_material_name,
        doping_percent,
    ):
        self.base_material = parse_alloy(base_material_name)
        self.doping_material = parse_alloy(doping_material_name)
        self.doping_percent = doping_percent

    def calc_bandgap(
        self,
        semiconductor_data='../data/semiconductors.csv'
    ):

        df = pd.read_csv(semiconductor_data)            
        df_mat = df.set_index('Material')

        if len(self.base_material) == 2:
            comp1 = self.base_material.keys()[0]
            comp2 = self.base_material.keys()[1]

            if comp1 not in df_mat.index:
                print(f"{comp1} not in database {df_mat.index}")
            if comp2 not in df_mat.index:
                print(f"{comp2} not in database {df_mat.index}")

            if comp1 in df_mat.index and comp2 in df_mat.index:
                eg1 = df_mat.loc[comp1, 'Bandgap (Eg)']
                eg2 = df_mat.loc[comp2, 'Bandgap (Eg)']

                return vegards_law(eg1, eg2, self.base_material.values[1])



class PN_junction:
    def __init__(
        self,
        n_type_material,
        p_type_material,
        n_type_dims, 
        p_type_dims,
    ):
        self.n_type_material = n_type_material
        self.p_type_material = p_type_material
        self.n_type_dims = n_type_dims
        self.p_type_dims = p_type_dims

    def calc_diffusion_length():

    def calc_current(self, voltage):

class SolarCell:
    def __init__(
        self, 
        junctions
    ):
        pass