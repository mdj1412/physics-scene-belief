"""Plot catalog-paired contrasts for the preregistered endpoints and spatial-transfer sensitivity."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
score=json.loads((here/"score48.json").read_text())
transfer=json.loads((here/"transfer48.json").read_text())
fig,(ax,bx)=plt.subplots(1,2,figsize=(10.6,3.65),layout="constrained")
colors=["#18637b","#a64931"]
ax.axvline(0,color="#888888",lw=1)
for j,row in enumerate([*score["primary"],score["preload_minus_none"]]):
    m,lo,hi=np.array(row)*1000
    ax.errorbar(m,j,xerr=[[m-lo],[hi-m]],c=colors[j%2],fmt="o",capsize=5)
ax.set(yticks=[0,1,2],yticklabels=["Always original","Fixed probe","No extra readout"],ylim=(2.5,-.5),
       xlabel="Query-site minus comparator Brier (×1,000)",xlim=(-12.7,.8))
ax.set_title("A   Value of one local preload",loc="left",fontsize=13)
bx.axhline(0,color="#888888",lw=1)
q=np.array(transfer["transfer_probability"])
for j,label in enumerate(["Frozen fixed-site calibration","Fixed site, assumes shared state"]):
    m,lo,hi=(np.array(transfer["local_minus_fixed_pointwise_95"])[:,j,:]*1000).T
    bx.plot(q,m,"o-",label=label,c=colors[j],lw=1.7,ms=4)
    bx.fill_between(q,lo,hi,color=colors[j],alpha=.12)
bx.set(xlabel="Probability of copying the state between sites",ylabel="Local minus fixed Brier (×1,000)",xticks=q)
bx.set_title("B   Where location matters",loc="left",fontsize=13)
bx.legend(frameon=False,fontsize=8,loc="lower right")
for plot in (ax,bx):
    plot.spines[["top","right"]].set_visible(False)
fig.savefig(here/"preload_v48.png",dpi=180,bbox_inches="tight")
