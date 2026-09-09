import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from textwrap import wrap

# TODO-TD: scale diffusion lengths to m instead of um

# SR NIEL M.J. Boschini, P.G. Rancoita and M. Tacconi (2014), SR-NIEL–7 Calculator: Screened Relativistic (SR) Treatment for NIEL Dose, Nuclear and Electronic Stopping Power Calculator (version 11.1); website https://www.sr-niel.org/ accessed on [year, month ].

# TODO-TD: add/drop 0 DDD case
# GaAs,electron,1.,0.,1E-16, 1.3E-8, 2.,4.8e16,1.32,8.0,0.15
# GaAs,electron,3.,0.,1E-16, 1.3E-8, 2.,4.8e16,1.32,8.0,0.15

# TODO-TD: should we do this in logspace?

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
    p0=None # TODO-TD: use kwargs and unpack
):
    """

    Returns 
    -------
    popt

    perr
    """
    if p0 is None:
        p0 = [1e4, 1e-16]

    popt, pcov = curve_fit(
        diffusion_length_fluence,
        fluences,
        diffusion_lengths,
        p0=p0,
        bounds=([0, 0], [np.inf, np.inf]), # Both Positive
        max_nfev=1e6,
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
        do_line=False,
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
                    if do_line:
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
                                p0=[1e15, 1e-14],
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
    plot_grouped(a1, 'fluence (e/cm^2)', 'L_n', True, do_model_fit=True)
    a1.set_xlabel('Fluence $(n/{cm}^{2})$')
    a1.set_ylabel('${L}_{n}$ (um)')
    a1.set_xscale('log')
    a1.set_yscale('log')
    plt.tight_layout()

    a2.grid(ls='--', which='both')
    plot_grouped(a2, 'DDD', 'L_n', do_lin_fit=True)
    a2.set_xlabel('DDD (MeV/g)')
    a2.set_xscale('log')
    a2.set_yscale('log')
    f1.suptitle('${L}_{n}$ vs Fluence & DDD in GaAs')

    plt.tight_layout()

    f1.savefig('./data/L_n_fluence_DDD.png')
    plt.close(f1)

    f2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    a1.grid(ls='--', which='both')
    plot_grouped(a1, 'fluence (e/cm^2)', 'L_p', True)
    a1.set_xlabel('Fluence $(n/{cm}^{2})$')
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.set_ylabel('${L}_{p}$ (um)')
    plt.tight_layout()

    a2.set_xlabel('DDD (MeV/g)')
    a2.set_xscale('log')
    a2.set_yscale('log')
    a2.grid(ls='--', which='both')
    plot_grouped(a2, 'DDD', 'L_p')
    f2.suptitle('${L}_{p}$ vs Fluence & DDD in GaAs')
    plt.tight_layout()
    f2.savefig('./data/L_p_fluence_DDD.png')
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
    f3.savefig('./data/I02_DDD.png')
    plt.close(f3)

    # TODO-TD: plot  model line in range for each particle energy

if __name__ == '__main__':
    main()