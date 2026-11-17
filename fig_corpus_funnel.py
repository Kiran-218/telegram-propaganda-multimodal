# fig_corpus_funnel.py  (corrected)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family':'serif','font.size':10})

def funnel(stages, fname, w=6.4, bw=2.1):
    fig, ax = plt.subplots(figsize=(w,2.4))
    n = len(stages)
    # give the end boxes room: first centre at bw/2 + pad, last at 10 - bw/2 - pad
    pad = 0.3
    left  = bw/2 + pad
    right = 10 - bw/2 - pad
    xs = [left + i*((right-left)/(n-1)) for i in range(n)]
    ax.set_xlim(0, 10)          # full canvas
    ax.set_ylim(0, 3)
    ax.axis('off')
    for i,(x,(num,lab)) in enumerate(zip(xs,stages)):
        ax.add_patch(FancyBboxPatch((x-bw/2,1.05),bw,1.0,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            fc="white",ec="#0009B2",lw=1.4))
        ax.text(x,1.74,num,ha='center',va='center',fontsize=11,fontweight='bold',color="#222")
        ax.text(x,1.32,lab,ha='center',va='center',fontsize=7.4,color="#666")
        if i<n-1:
            ax.add_patch(FancyArrowPatch((x+bw/2,1.55),(xs[i+1]-bw/2,1.55),
                arrowstyle='-|>',mutation_scale=12,lw=1.1,color="#666"))
    fig.savefig(fname, bbox_inches='tight', dpi = 300)

funnel([
    ("55,423","total posts"),
    ("22,650","posts with images"),
    ("9,054","valid pairs\n(92 channels)"),
], "fig_corpus_funnel.png", w=6.4, bw=2.1)