# fig_hamming_distribution.py  — uses your reuse_distance_distribution.csv values
import matplotlib.pyplot as plt

d = [0,2,4,6,8,10]                 # replace with your actual distances if different
counts = [407,90,12,3,2,13]        # from reuse_distance_distribution.csv
plt.rcParams.update({'font.family':'serif','font.size':10})
fig, ax = plt.subplots(figsize=(6.0,3.4))
colors = ["#002AB2" if x<=6 else '#BBBBBB' for x in d]
bars = ax.bar([str(x) for x in d], counts, color=colors, edgecolor='white', width=0.7)
for b,c in zip(bars,counts):
    ax.text(b.get_x()+b.get_width()/2, c+6, str(c), ha='center', va='bottom', fontsize=8.5)
ax.axvline(3.5, color="#D51900", ls='--', lw=1.3)
ax.text(3.5, max(counts)*0.82, ' threshold ≤ 6', color='#D55E00', fontsize=8.5, ha='left')
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.set_xlabel('Hamming distance between perceptual hashes')
ax.set_ylabel('Cross-channel pair count'); ax.set_ylim(0, max(counts)*1.12)
fig.savefig('fig_hamming_distribution.png', bbox_inches='tight', dpi = 300)