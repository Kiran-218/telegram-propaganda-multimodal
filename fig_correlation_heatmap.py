# fig_correlation_heatmap.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

M = pd.read_csv('spearman_matrix_rho.csv', index_col=0)   # your file

# clean, human-readable labels (order preserved from the file)
label_map = {
    'mismatch_mean':      'mismatch (mean)',
    'mismatch_std':       'mismatch (std)',
    'participants_count': 'audience size',
    'channel_age_days':   'channel age',
    'forward_rate':       'forward rate',
    'posts_per_day':      'posts/day',
    'mean_views':         'per-post views',
    'reuse_rate_cross':   'reuse rate',
    'n_reuse_partners':   'reuse partners',
}
labels = [label_map.get(c, c) for c in M.columns]

plt.rcParams.update({'font.family':'serif','font.size':9})
fig, ax = plt.subplots(figsize=(6.6,5.8))

im = ax.imshow(M.values, cmap='RdBu_r', vmin=-1, vmax=1)

# annotate each cell
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.8,
                color='white' if abs(v) > 0.6 else '#222')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)

# thin white gridlines between cells
ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.5)
ax.tick_params(which='minor', length=0)
for s in ax.spines.values():
    s.set_visible(False)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(r'Spearman $\rho$', fontsize=8)
cb.ax.tick_params(labelsize=7)

fig.savefig('fig_correlation_heatmap.png', bbox_inches='tight', dpi = 300)