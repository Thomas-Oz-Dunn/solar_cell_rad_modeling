import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from textwrap import wrap
from pathlib import Path
import scipy.constants as const

# TODO-TD: scale diffusion lengths to m instead of um

# SR NIEL M.J. Boschini, P.G. Rancoita and M. Tacconi (2014), SR-NIEL–7 Calculator: Screened Relativistic (SR) Treatment for NIEL Dose, Nuclear and Electronic Stopping Power Calculator (version 11.1); website https://www.sr-niel.org/ accessed on [year, month ].

# TODO-TD: add/drop 0 DDD cases
# GaAs,electron,1.,0.,1E-16, 1.3E-8, 2.,4.8e16,1.32,8.0,0.15
# GaAs,electron,3.,0.,1E-16, 1.3E-8, 2.,4.8e16,1.32,8.0,0.15

# TODO-TD: should we do this in logspace?

# TODO-TD: Spectral EQE plots?

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

def fit_diffusion_length_fluence(
    fluences, 
    diffusion_lengths,
    p0=None, # TODO-TD: use kwargs and unpack
    log_space=True,
    max_n_steps=1e12,
):
    """

    Returns 
    -------
    popt

    perr
    """
    if p0 is None:
        p0 = [1e4, 1e-16]

    if log_space:
        def model(fluence, L_0, K_L):
            return np.log10(diffusion_length_fluence(fluence, L_0, K_L))

        ydata = np.log10(diffusion_lengths)
    else:
        model = diffusion_length_fluence
        ydata = diffusion_lengths

    popt, pcov = curve_fit(
        model,
        fluences,
        ydata,
        p0=p0,
        bounds=([0, 0], [np.inf, np.inf]), # Both Positive
        max_nfev=max_n_steps,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def linear_log_log_fit(x, y):
    # Linear fit in log space

    logy = np.log10(y)
    results = np.polyfit(
        np.log10(x), 
        logy, 
        1,
        full=True,
    )
    slope, intercept = results[0]
    # Residual or Sum of Square Error (SSE)
    SSE = results[1][0]

    # Determining the Sum of Square Total (SST)
    # the squared differences between the observed dependent variable and its mean
    diffs = logy - logy.mean()
    square_diff = diffs ** 2
    SST = square_diff.sum()

    R2 = 1 - SSE/SST 
    return slope, intercept, R2

# TODO-TD: add flags for loglog vs linlin plotting
def main():
    COLORS = {
        'electron': "#1f26b4",
        'proton': "#d60303"
    }
    MeV_MARKERS = {1: 'o', 3: 's', 10: 'd'}  # by energy (MeV)

    df = pd.read_csv('./data/SalzbergerEtal2018.csv')
    df.columns = [c.strip() for c in df.columns]
    print("Rows:", len(df))
    print("Materials:", sorted(df['material'].unique()))
    print("Particle types:", sorted(df['particle type'].unique()))
    print("Energies (MeV):", sorted(df['energy (MeV)'].unique()))

    dfNIEL = pd.read_csv('./data/SRNIEL_TABLE.csv')
    dfNIEL.columns = [c.strip() for c in dfNIEL.columns]

    df2 = df.merge(dfNIEL, on=['material', 'particle type', 'energy (MeV)'], how='left')
    df2['DDD'] = df2['fluence (e/cm^2)'].astype(float) * df2['NIEL (MeV cm^2/g)'].astype(float) 

    def plot_grouped(
        ax, 
        x_col,
        y_col,
        plot_line=False,
        do_lin_fit=False,
        do_model_fit=False,
    ):
        """
        Combine same plot technique
        """
        for ptype in df2['particle type'].unique():
            sub_df = df2[df2['particle type'] == ptype]
            for energy in sorted(sub_df['energy (MeV)'].unique()):
                sub_e = sub_df[sub_df['energy (MeV)'] == energy]
                for material in sub_e['material'].unique():
                    sub2df = sub_e[sub_e['material'] == material].sort_values(x_col)
                    # TODO-TD: use kwargs
                    if plot_line:
                        ax.plot(
                            sub2df[x_col],
                            sub2df[y_col],
                            marker=MeV_MARKERS.get(energy, 'o'),
                            color=COLORS.get(ptype, 'gray'),
                            label=f"{energy} MeV {ptype}",
                        )
                    else:
                        ax.scatter(
                            sub2df[x_col],
                            sub2df[y_col],
                            marker=MeV_MARKERS.get(energy, 'o'),
                            color=COLORS.get(ptype, 'gray'),
                            label=f"{energy} MeV {ptype}",
                        )
                        
                    if do_model_fit and len(sub2df[x_col]) >= 2:
                        # For each particle energy
                        if ptype == 'electron':
                            popt, perr = fit_diffusion_length_fluence(
                                sub2df[x_col], 
                                sub2df[y_col],
                                p0=[10, 1e-20],
                            )
                        else:
                            popt, perr = fit_diffusion_length_fluence(
                                sub2df[x_col], 
                                sub2df[y_col],
                            )

                        y_fit_2 = diffusion_length_fluence(
                            sub2df[x_col], 
                            popt[0], 
                            popt[1],
                        )
                        ax.plot(
                            sub2df[x_col],
                            y_fit_2,
                            color=COLORS.get(ptype, 'gray'),
                            ls='--',
                            alpha=0.8,
                            label=f'1/L^2 fit L_0={popt[0]:0.4g} K_L={popt[1]:0.4g} perr={perr[0]:0.4g} {perr[1]:0.4g}'
                        )

        x = df2[x_col]
        y = df2[y_col]
        if do_lin_fit and len(x) >= 2:

            # L_0 sqrt(1 / (1 +- L_0^2 K_L phi))?

            # mask out any NaNs before fitting
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 2:
                slope, intercept, R2 = linear_log_log_fit(x[mask], y[mask])
                x_fit = np.linspace(
                    np.log10(x[mask]).min(), 
                    np.log10(x[mask]).max(), 
                    100,
                )
                y_fit = slope * x_fit + intercept
                equa = f'log10({y_col}) = {slope:0.4g} * log10({x_col}) + {intercept:0.4g} (R^2 = {R2:0.4g})'
                wrapped = '\n'.join(wrap(equa, 20))
                ax.plot(
                    10 ** (x_fit), 
                    10 ** (y_fit),
                    color='k',
                    ls='--',
                    linewidth=1,
                    alpha=0.8,
                    label=wrapped,
                )
        ax.legend()

    f1, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    a1.grid(ls='--', which='both')
    plot_grouped(a1, 'fluence (e/cm^2)', 'L_n', True, do_model_fit=False)
    a1.set_xlabel('Fluence $(n/{cm}^{2})$')
    a1.set_ylabel('${L}_{n}$ (um)')
    a1.set_xscale('log')
    # a1.set_yscale('log')
    plt.tight_layout()

    a2.grid(ls='--', which='both')
    plot_grouped(a2, 'DDD', 'L_n', do_lin_fit=False, do_model_fit=True)
    a2.set_xlabel('DDD (MeV/g)')
    a2.set_xscale('log')
    # a2.set_yscale('log')
    f1.suptitle('${L}_{n}$ vs Fluence & DDD in GaAs')

    plt.tight_layout()

    f1.savefig('./processing/L_n_fluence_DDD.png')
    plt.close(f1)

    f2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    a1.grid(ls='--', which='both')
    plot_grouped(a1, 'fluence (e/cm^2)', 'L_p', True)
    a1.set_xlabel('Fluence $(n/{cm}^{2})$')
    a1.set_xscale('log')
    # a1.set_yscale('log')
    a1.set_ylabel('${L}_{p}$ (um)')
    plt.tight_layout()

    a2.set_xlabel('DDD (MeV/g)')
    a2.set_xscale('log')
    # a2.set_yscale('log')
    a2.grid(ls='--', which='both')
    plot_grouped(a2, 'DDD', 'L_p')
    f2.suptitle('${L}_{p}$ vs Fluence & DDD in GaAs')
    plt.tight_layout()
    f2.savefig('./processing/L_p_fluence_DDD.png')
    plt.close(f2)

    def I02_DDD_model(I_02_0, DDD, DDD_0):
        return I_02_0 * (1 + DDD/DDD_0)

    f3, a1 = plt.subplots(1, 1, figsize=(6, 6))
    a1.set_xlabel('DDD (MeV/g)')
    a1.set_ylabel('${I}_{02}$ $(mA/{cm}^{2})$')
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.grid(ls='--', which='both')
    plot_grouped(a1, 'DDD', 'I_02', True)
    a1.set_title('${I}_{02}$ vs DDD in GaAs')
    f3.savefig('./processing/I02_DDD.png')
    plt.close(f3)

    f_eqe, a_eqe = plt.subplots(1, 1, figsize=(6, 6))

    # Excess carriers Δn ≈ 1e14 cm⁻³ (1e20 m⁻³), 
    # τ_eff ≈ 10 ns (1e-8 s)
    thickness1 = 2e-3
    recomb_rate_SI = 1e20 / 1e-8  # 1e28 m⁻³ s⁻¹
    a_eqe = plot_spectral_eqe(
        thickness1, 
        recomb_rate_SI, 
        'InP',
        a_eqe,
    )
    thickness2 = 2e-2
    a_eqe = plot_spectral_eqe(
        thickness2, 
        recomb_rate_SI, 
        'InP',
        a_eqe,
    )
    thickness2 = 2e-1
    a_eqe = plot_spectral_eqe(
        thickness2, 
        recomb_rate_SI, 
        'InP',
        a_eqe,
    )
    a_eqe.legend()
    f_eqe.savefig('./processing/InP_EQE.png')

    print('Plots can be found in ./processing/')

if __name__ == '__main__':
    main()