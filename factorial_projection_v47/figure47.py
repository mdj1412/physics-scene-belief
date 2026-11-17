"""Two views of the same matched experiment. Inputs are server-scored catalog summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
score=json.loads((root/"score47.json").read_text())
sensitivity=json.loads((root/"polarity47.json").read_text())
fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
blue,red="#246682","#ad4231"
ax.axhline(0,color="#757575",lw=.9)
for i,label in enumerate(["Low alignment  (0.1)","High alignment (0.9)"]):
    point,lo,hi=np.asarray(score["forced_by_factor"])[i].T*1000
    x=np.array([0,1])+(-.03 if i==0 else .03)
    ax.errorbar(x,point,yerr=[point-lo,hi-point],fmt="o-",capsize=4,color=[blue,red][i],label=label)
ax.set(xticks=[0,1],xticklabels=["Inert","Load-bearing"],xlim=(-.33,1.33),ylabel="Projected − original Brier (×1,000)")
ax.set_title("A   Effect of removing the source axis",loc="left",fontsize=12)
ax.legend(frameon=False,fontsize=8,loc="upper left")
bx.axvline(0,color="#757575",lw=.9)
for j,key in enumerate(("recorded_interaction","random_sign_interaction","reversed_interaction")):
    est,lo,hi=np.asarray(sensitivity[key])*1000
    bx.errorbar(est,j,xerr=[[est-lo],[hi-est]],fmt="o",capsize=4,c=[red,"#474747",blue][j])
bx.set(yticks=[0,1,2],yticklabels=["As recorded","Random polarity","Reversed"],ylim=(2.5,-.5),xlabel="Alignment × coupling interaction (×1,000)")
bx.set_title("B   Optical and mechanical polarity",loc="left",fontsize=12)
for a in (ax,bx):
    a.spines[["top","right"]].set_visible(False)
fig.savefig(root/"factorial_v47.png",dpi=190,bbox_inches="tight")
