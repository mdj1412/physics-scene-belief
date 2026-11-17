"""Display two-shape effects and the distribution shift in the two active position bins."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from geometry46 import shape_profile

ROOT=Path(__file__).resolve().parent
def read(k):return json.loads((ROOT/(k+".json")).read_text())
colors=["#246582","#a34b30"]
a,b=read("score46"),read("score46_bullet")
src,tgt=read("router46")["visual"],read("transfer46")["visual"]
fig,ax=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={"width_ratios":[1,1.32]},layout="constrained")
ax[0].axvline(0,c="#888",lw=.8)
for i in range(2):
    for j,(data,mk) in enumerate([(a,"o"),(b,"s")]):
        m,lo,hi=1000*np.array(data["family_contrasts"])[i,0]
        ax[0].errorbar(m,i+(j-.5)*.18,xerr=[[m-lo],[hi-m]],fmt=mk,c=colors[i],lw=1.7,capsize=3,ms=7,
                       markerfacecolor=colors[i] if j==0 else "white")
ax[0].set(yticks=[0,1],yticklabels=["Terrace","Dimple"],ylim=(1.62,-.68),xlim=(-.6,14.9),
          xlabel="Joint − always-original Brier (×1,000)")
ax[0].set_title("A   Cost of routing by geometry",loc="left",pad=12)
ax[0].legend(handles=[Line2D([],[],marker="o",ls="none",color="#444",label="Finite pressure"),
     Line2D([],[],marker="s",mfc="white",ls="none",color="#444",label="Rigid PyBullet")],frameon=False,loc="lower right",fontsize=9)
ax[1].axhline(0,c="#666",lw=.8)
target=np.array(tgt["target_raw_D"]); weight=np.array(tgt["count"],float)
for i,k in enumerate([3,4]):
    x=np.arange(4)+5*i
    risk=1000*np.array(src["risk_joint"])[k]
    area=weight[k]/weight[k].sum()
    ax[1].plot(x,risk,"x--",c="#494949",lw=1,ms=6)
    ax[1].scatter(x,target[k]*1000,s=15+210*area,c=colors[i],alpha=.88,edgecolor="white",zorder=4)
    ax[1].text(x.mean(),32.5,f"{[30.5,31][i]} mm",ha="center",color=colors[i])
ax[1].set(xticks=[0,1,2,3,5,6,7,8],xticklabels=["Q1","Q2","Q3","Q4"]*2,ylim=(-10,36),xlim=(-.55,8.6),
          ylabel="Projected − original Brier (×1,000)",xlabel="Within-position source variance quartile")
ax[1].set_title("B   Source rule and realized target loss",loc="left",pad=12)
ax[1].legend(handles=[Line2D([],[],marker="x",ls="--",lw=1,c="#494949",label="Source estimate"),
   Line2D([],[],marker="o",ls="none",c="#467f88",label="Target (size = frequency)")],frameon=False,loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8)
for a0 in ax:a0.spines[["top","right"]].set_visible(False)
fig.savefig(ROOT/"normal_variance_v46.png",dpi=180,bbox_inches="tight")
fig,axes=plt.subplots(2,2,figsize=(9,4.5),layout="constrained")
x=np.linspace(9,40,300)
for i,shape in enumerate(["terrace","dimple"]):
    cam=np.load(ROOT/"camera46"/"panel_00"/f"pixels_{shape}_example.npz")["frames"].reshape(3,32,96)
    axes[i,0].imshow(np.concatenate([c for c in cam],axis=1),cmap="gray",vmin=0,vmax=255)
    axes[i,0].set_axis_off()
    axes[i,0].set_title(f"{shape.title()} · pre-impact sequence",fontsize=10,loc="left")
    for variant,ls in [("hollow","--"),("base","-"),("bulge",":")]:
        z=shape_profile(x,shape,variant)[0]
        axes[i,1].plot(x,z,ls,label=variant,lw=1.7)
    for h in (19,31):axes[i,1].axvspan(h-2,h+2,color="#333",alpha=.08)
    axes[i,1].set(xlim=(9,40),ylabel="Profile (mm)")
    axes[i,1].spines[["top","right"]].set_visible(False)
axes[0,1].legend(frameon=False,ncol=3,fontsize=8)
axes[1,1].set_xlabel("Contact coordinate (mm)")
fig.savefig(ROOT/"families_v46.png",dpi=165,bbox_inches="tight")
