"""Draw the two prespecified simulator comparisons from their archived numeric results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

root=Path(__file__).resolve().parent
law=json.loads((root/"score.json").read_text())["primary"]
light=json.loads((root.parent/"paired_light_sham_v67b"/"results67.json").read_text())
fig,(left,right)=plt.subplots(1,2,figsize=(11.2,3.9),gridspec_kw={"width_ratios":[.95,1.05]})
cols=("#176879","#aa451b")
for j,key in enumerate(("strengthening","weakening")):
    res=law[key]
    arr=np.array([*res["family_soft"],res["pooled_rescue"]])*1000
    left.errorbar(arr[:,0],np.arange(3)-.09+.18*j,xerr=np.vstack([arr[:,0]-arr[:,1],arr[:,2]-arr[:,0]]),
                  fmt=("o","s")[j],ms=5,capsize=3,color=cols[j],label=key.capitalize())
left.axvline(0,lw=.8,color=".35")
left.set_yticks([0,1,2],["Wendland\nsoft − full","Dispersive\nsoft − full","Pooled\nrestored − dropped"])
left.set_ylim(2.65,-.55)
left.set_xlabel("Difference in held-out Brier loss (×10⁻³)")
left.set_title("A    Contact law and geometry",loc="left",fontweight="bold")
left.legend(frameon=False,loc="lower right",fontsize=8)
cells=("no_added","marker","delay_raw","both_raw","both_corrected")
for j,arm in enumerate(("Same-light sham","Two lights")):
    arr=np.array([light["terminal_ci"][j][light["cases"].index(k)] for k in cells])*100
    right.errorbar(np.arange(5)-.1+.2*j,arr[:,0],yerr=[arr[:,0]-arr[:,1],arr[:,2]-arr[:,0]],
                   capsize=3,fmt=("o","s")[j],ms=5,color=cols[j],label=arm)
right.set_xticks(range(5),["No added\nfault","Marker","Delay","Marker\n+ delay","Both after\ncorrection"])
right.set_ylabel("Final purchase disagreement (%)")
right.set_ylim(bottom=0)
right.set_title("B    Timing, marker identity and purchase",loc="left",fontweight="bold")
right.legend(frameon=False,loc="upper left",fontsize=8)
for ax in (left,right):
    ax.spines[["right","top"]].set_visible(False)
    ax.tick_params(length=3,color=".45",labelsize=9)
left.xaxis.set_major_locator(MaxNLocator(6))
fig.tight_layout(w_pad=2)
for ext in ("pdf","png"):
    fig.savefig(root/f"v67_summary.{ext}",bbox_inches="tight",dpi=180)
