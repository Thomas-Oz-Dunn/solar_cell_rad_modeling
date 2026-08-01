import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data\semiconductors.csv')

# TODO-TD: Use vegards_law to interpolate alloys
# Band parameters for III–V compound semiconductors and their alloys


df['Bandgap (Eg)'] = df['Bandgap (Eg)'].str.replace(' eV', '').astype(float)

color_dict = {
  'Diamond (FCC)': 'red', 
  'Zinc blende (FCC)': 'blue', 
  'Wurtzite': 'green',
}

plt.figure(figsize=(8, 6))

for structure, color in color_dict.items():
    subset = df[df['Crystal Structure'] == structure]
    plt.scatter(
        subset['a (A)'],
        subset['Bandgap (Eg)'],
        color=color,
        s=50,
        label=structure
    )


for i, row in df.iterrows():
    plt.annotate(
      row['Material'], 
      (row['a (A)'], row['Bandgap (Eg)']),
      textcoords="offset points", 
      xytext=(0, 10)
    )

plt.legend(title="Crystal Structure")
plt.title('Semiconductor Bandgap vs. Lattice Constant (a)')
plt.xlabel('Lattice Constant a (A)')
plt.ylabel('Bandgap $E_g$ (eV)')
plt.grid(ls='--', alpha=0.3)

plt.tight_layout()
plt.show()