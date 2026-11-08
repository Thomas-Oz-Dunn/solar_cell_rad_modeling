import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosci.utils import eV_to_joules, energy_to_wavelength, eqe_dynamic_alpha_InP_300K, vegards_law

def plot_spectral_eqe(thickness, recomb_rate, material, ax=None):
    # TODO-TD: cut off frequency based on thickness?
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    df = pd.read_csv(Path('./data/wmo.csv'), sep=r'\s+')

    material_df = pd.read_csv(Path('./data/semiconductors.csv'))
    material_df['Bandgap (Eg)'] = (
        material_df['Bandgap (Eg)']
        .str.replace(' eV', '')
        .astype(float)
    )
    eg_joules = eV_to_joules(
        material_df.loc[
            material_df['Material'] == material,
            'Bandgap (Eg)'
        ]
    )
    wavelength_m_eg = energy_to_wavelength(eg_joules.iloc[0])

    wavelengths_nm = df['nm'].to_numpy()
    wavelengths_m = wavelengths_nm * 1e-9
    power_density = df['W/sm/nm'].to_numpy()

    print(wavelength_m_eg)
    eqe_vals = eqe_dynamic_alpha_InP_300K(
        thickness,
        wavelengths_m,
        recomb_rate,
        power_density,
    )
    wavelengths_nm = np.where(
        wavelengths_m <= wavelength_m_eg, 
        wavelengths_nm, 
        None,
    )
    print(wavelengths_nm)
    eqe_vals = np.where(
        wavelengths_m <= wavelength_m_eg, 
        eqe_vals, 
        None,
    )

    ax.plot(
        wavelengths_nm, 
        eqe_vals, 
        label=f'thickness: {thickness} R: {recomb_rate}',
    )
    plt.grid(ls='--')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('EQE')
    ax.set_title(f'{material} Spectral EQE')

    return ax

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
