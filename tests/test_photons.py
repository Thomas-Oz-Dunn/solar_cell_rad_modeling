import pytest
import rosci

@pytest.mark.parametrize("wavelength", [
    1e-9,    # 1 nm (X-rays / EUV)
    500e-9,  # 500 nm (Visible green light)
    10.6e-6, # 10.6 µm (Infrared)
    1e-2     # 1 cm (Microwaves)
])
def test_wavelength_energy_roundtrip(wavelength):
    energy = rosci.wavelength_to_energy(wavelength)
    returned_wavelength = rosci.energy_to_wavelength(energy)
    
    assert returned_wavelength == pytest.approx(wavelength, rel=1e-12)


@pytest.mark.parametrize("energy", [
    1e-18,   # High energy photon
    4e-19,   # Visible light 
    1e-22    # Far infrared photon
])
def test_energy_wavelength_roundtrip(energy):
    wavelength = rosci.energy_to_wavelength(energy)
    returned_energy = rosci.wavelength_to_energy(wavelength)
    
    assert returned_energy == pytest.approx(energy, rel=1e-12)