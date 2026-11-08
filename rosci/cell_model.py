from rosci.utils import vegards_law


semiconductor_data='../data/semiconductors.csv'

class Semiconductor:
    def __init__(
        self,
        base_material,
        doping_material,
        doping_percent,
    ):
        self.base_material = base_material
        self.doping_material = doping_material
        self.doping_percent = doping_percent

    def calc_bandgap(self):
        vegards_law()


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