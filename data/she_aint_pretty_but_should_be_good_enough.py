import pandas as pd
import matplotlib.pyplot as plt

# TODO-TD: scale diffusion lengths to m instead of um

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

    df2 = df.merge(dfNIEL, on=['particle type', 'energy (MeV)'], how='left')
    df2['DDD'] = df2['fluence (e/cm^2)'].astype(float) * df2['NIEL (MeV cm^s/g)'].astype(float) 

    def plot_grouped(
        ax, 
        x_col,
        y_col,
        do_line=False
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
                            label=f"{ptype}, {energy} MeV, {material}",
                        )
                    else:
                        ax.scatter(
                            sub2df[x_col],
                            sub2df[y_col],
                            marker=MeV_MARKERS.get(energy, 'o'),
                            color=COLORS.get(ptype, 'gray'),
                            label=f"{ptype}, {energy} MeV, {material}",
                        )
        ax.legend(fontsize='small')

    f1, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_grouped(a1, 'fluence (e/cm^2)', 'L_n', True)
    a1.grid(ls='--')
    a1.set_xlabel('Fluence (e/cm^2)')
    a1.set_ylabel('L_n (um)')
    a1.set_xscale('log')
    a1.set_yscale('log')
    plt.tight_layout()

    plot_grouped(a2, 'DDD', 'L_n')
    a2.set_xlabel('DDD (MeV/g)')
    a2.set_xscale('log')
    a2.set_yscale('log')
    a2.grid(ls='--')
    f1.suptitle('L_n vs Fluence & DDD in GaAs')
    plt.tight_layout()

    f1.savefig('./data/L_n_fluence_DDD.png')
    plt.close(f1)

    f2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_grouped(a1, 'fluence (e/cm^2)', 'L_p', True)
    a1.grid(ls='--')
    a1.set_xlabel('Fluence (e/cm^2)')
    a1.set_xscale('log')
    a1.set_yscale('log')
    a2.set_ylabel('L_p (um)')
    plt.tight_layout()

    plot_grouped(a2, 'DDD', 'L_p')
    a2.set_xlabel('DDD (MeV/g)')
    a2.set_xscale('log')
    a2.set_yscale('log')
    a2.grid(ls='--')
    f2.suptitle('L_p vs Fluence & DDD in GaAs')
    plt.tight_layout()
    f2.savefig('./data/L_p_fluence_DDD.png')
    plt.close(f2)

    def I02_DDD(I_02_0, DDD, DDD_0):
        return I_02_0 * (1 + DDD/DDD_0)

    f3, a1 = plt.subplots(1, 1, figsize=(6, 6))
    plot_grouped(a1, 'DDD', 'I_02', True)
    a1.set_xlabel('DDD (MeV/g)')
    a1.set_ylabel('I_02')
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.grid(ls='--')
    a1.set_title('I_02 vs DDD in GaAs')
    f3.savefig('./data/I02_DDD.png')
    plt.close(f3)

    # TODO-TD: plot  model line in range for each particle energy

if __name__ == '__main__':
    main()