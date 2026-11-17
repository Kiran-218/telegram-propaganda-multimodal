# fig_mismatch_distribution.py
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('eval_merged.csv')          # your file
ev = df[df['evaluable']==True]
prop = ev[ev['label']=='propagandistic']['mclip_mismatch'].values
neut = ev[ev['label']=='neutral']['mclip_mismatch'].values

plt.rcParams.update({'font.family':'serif','font.size':10})
fig, ax = plt.subplots(figsize=(6.2,3.6))
bins = np.linspace(0.60, 0.88, 34)
ax.hist(neut, bins=bins, density=True, alpha=0.55, color='#0072B2',
        label=f'Neutral (n={len(neut)})', edgecolor='white', linewidth=0.4)
ax.hist(prop, bins=bins, density=True, alpha=0.55, color='#D55E00',
        label=f'Propagandistic (n={len(prop)})', edgecolor='white', linewidth=0.4)
ax.axvline(neut.mean(), color="#0027B2", ls='--', lw=1.2)
ax.axvline(prop.mean(), color="#D50400", ls='--', lw=1.2)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.set_xlabel('Cross-modal mismatch score'); ax.set_ylabel('Density')
ax.legend(frameon=False, fontsize=9, loc='upper left')
diff = abs(prop.mean()-neut.mean()); sd = ev['mclip_mismatch'].std()
ax.annotate(f'means differ by {diff:.4f}\n(≈ {diff/sd:.2f} SD)',
            xy=(0.83,0.5), xycoords='axes fraction', fontsize=8.5,
            color='#666', ha='left', va='center')
fig.savefig('fig_mismatch_distribution.png', bbox_inches='tight', dpi = 300)