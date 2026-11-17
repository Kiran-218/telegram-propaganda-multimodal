# fig_probe_architecture.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family':'serif','font.size':10})
fig, ax = plt.subplots(figsize=(6.6,4.2))
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')

def box(x,y,w,h,text,ec="#0027B2",fs=8.5,bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle="round,pad=0.02,rounding_size=0.06",fc="white",ec=ec,lw=1.3))
    ax.text(x,y,text,ha='center',va='center',fontsize=fs,
            fontweight='bold' if bold else 'normal',color="#222")

def arrow(x1,y1,x2,y2,color="#666"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
        mutation_scale=11,lw=1.0,color=color))

box(5,9.2,3.4,0.9,"Post (image + caption)",ec="#666",bold=True)
box(2.6,7.4,3.0,0.9,"M-CLIP\nimage encoder")
box(7.4,7.4,3.0,0.9,"M-CLIP\ntext encoder")
arrow(4.2,8.75,3.0,7.9); arrow(5.8,8.75,7.0,7.9)
box(2.6,5.8,2.6,0.8,"image emb.\n(512-d)",ec="#666",fs=8)
box(7.4,5.8,2.6,0.8,"text emb.\n(512-d)",ec="#666",fs=8)
arrow(2.6,6.95,2.6,6.25); arrow(7.4,6.95,7.4,6.25)
box(5,4.1,7.2,0.85,"feature sets: image / text / concat (1024-d) / late fusion",ec="#047456",fs=8)
arrow(2.6,5.35,3.5,4.55); arrow(7.4,5.35,6.5,4.55)
box(5,2.5,5.4,0.85,"regularised logistic regression\n(nested CV, permutation test)")
arrow(5,3.65,5,2.95)
box(5,0.9,3.4,0.8,"propaganda probability",ec="#D50400",bold=True)
arrow(5,2.05,5,1.32)

fig.savefig('fig_probe_architecture.png', bbox_inches='tight', dpi = 300)