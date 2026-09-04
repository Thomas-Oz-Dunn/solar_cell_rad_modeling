import pandas as pd
import matplotlib.pyplot as plt

def main():
    COLORS = {
        'electron': "#1f26b4", 
        'proton': "#d60303"
    }
    MeV_MARKERS = {1: 'o', 3: 's', 10: 'd'}  # by energy (MeV)

    df = pd.read_csv('../data/SalzbergerEtal2018.csv', encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    print("Rows:", len(df))
    print("Materials:", sorted(df['material'].unique()))
    print("Particle types:", sorted(df['particle type'].unique()))
    print("Energies (MeV):", sorted(df['energy (MeV)'].unique()))

    dfNIEL = pd.read_csv('../data/SRNIEL_TABLE.csv', encoding='utf-8-sig')
    dfNIEL.columns = [c.strip() for c in dfNIEL.columns]

    # TODO-TD: convert fluence to DDD using NIEL lookup?
    df2 = df.merge(dfNIEL, on=['particle', 'energy (MeV)'], how='left')

    f, a = plt.subplot()
    for ptype in df['particle type'].unique():
        sub_df = df[df['particle type'] == ptype]
        for energy in sub_df['energy (MeV)'].unique():
            sub2df = sub_df[sub_df['energy (MeV)'] == energy]
            a.plot(
                sub2df['fluence (e/cm^2)'], 
                sub2df['L_n'],
                marker=MeV_MARKERS.get(energy, 'o'),
                color=COLORS.get(ptype, 'gray'),
                linestyle='-',
                label=f"{ptype}, {energy} MeV",
            )



    f, a = plt.subplot()
    for ptype in df['particle type'].unique():
        sub_df = df[df['particle type'] == ptype]
        for energy in sub_df['energy (MeV)'].unique():
            sub2df = sub_df[sub_df['energy (MeV)'] == energy]
            a.plot(
                sub2df['fluence (e/cm^2)'] * sub2df['NIEL (MeV cm^s/g)'], 
                sub2df['L_n'],
                marker=MeV_MARKERS.get(energy, 'o'),
                color=COLORS.get(ptype, 'gray'),
                linestyle='-',
                label=f"{ptype}, {energy} MeV",
            )

if __name__ == '__main__':
    main()