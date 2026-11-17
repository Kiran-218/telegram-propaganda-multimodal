# fig_llm_confusion.py
import numpy as np
import matplotlib.pyplot as plt

# rows = human gold, cols = LLM prediction
cm = np.array([[73, 42],    # human propagandistic
               [29, 201]])  # human neutral
row_labels = ['Propagandistic', 'Neutral']   # human
col_labels = ['Propagandistic', 'Neutral']   # LLM
row_tot = cm.sum(axis=1, keepdims=True)
pct = cm / row_tot * 100   # row-normalised %

plt.rcParams.update({'font.family':'serif','font.size':10})
fig, ax = plt.subplots(figsize=(4.6,4.0))

# light shading by row-normalised value (diagonal darker where model agrees)
shade = pct/100
ax.imshow(shade, cmap='Blues', vmin=0, vmax=1, alpha=0.55)

for i in range(2):
    for j in range(2):
        ax.text(j, i-0.10, f'{cm[i,j]}', ha='center', va='center',
                fontsize=15, fontweight='bold', color='#222')
        ax.text(j, i+0.18, f'{pct[i,j]:.1f}%', ha='center', va='center',
                fontsize=9, color='#555')

ax.set_xticks([0,1]); ax.set_xticklabels(col_labels)
ax.set_yticks([0,1]); ax.set_yticklabels(row_labels, rotation=90, va='center')
ax.set_xlabel('LLM prediction'); ax.set_ylabel('Human gold label')
ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
ax.set_xticks(np.arange(-.5,2,1), minor=True)
ax.set_yticks(np.arange(-.5,2,1), minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='minor', length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.savefig('fig_llm_confusion.png', bbox_inches='tight', dpi = 300)