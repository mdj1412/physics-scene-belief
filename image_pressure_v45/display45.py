"""Render paired Brier contrasts. Horizontal intervals are simultaneous within each family."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
s=json.loads((root/"score45.json").read_text())["source_axis"]
fig,axes=plt.subplots(1,2,figsize=(10.4,4.0),layout="constrained")
for ax in axes:
    ax.axvline(0,color="#777",lw=.9)
    ax.spines[["right","top","left"]].set_visible(False)
    ax.grid(axis="x",alpha=.13)
    ax.tick_params(axis="y",length=0)
def dot(ax,row,values,col):
    m,lo,hi=np.array(values)*1000
    ax.errorbar(m,row,xerr=[[m-lo],[hi-m]],fmt="o",c=col,lw=2,capsize=4)
a,b=axes; colors=("#245c83","#b04e29")
a.set_title("A   Does image-based routing help?",loc="left",pad=16)
for y,d in enumerate(s["primary_comparisons"]):
    dot(a,y,d,"#245c83")
a.set(yticks=range(3),yticklabels=["vs always-original","vs oracle position","vs nominal height"],ylim=(2.7,-.75),
      xlabel="Visual − comparator Brier (×1,000)")
b.set_title("B   What changes with pressure area?",loc="left",pad=16)
for y,d in enumerate(s["pressure_contrasts"]):
    dot(b,y,d,colors[y] if y<2 else "#4f5257")
b.set(yticks=range(3),yticklabels=["Fold","Ripple","Fold − ripple"],ylim=(2.7,-.75),
      xlabel="Change in projection cost, broad − narrow (×1,000)")
for ax in axes:
    ax.margins(x=.09)
fig.savefig(root/"image_pressure_v45.png",dpi=180,bbox_inches="tight")
