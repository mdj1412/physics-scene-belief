"""Standalone publication figure: catalog-wise paired contrasts and analytic twin floor."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent


def add_points(ax, values, names, colors):
    pos=np.arange(len(names))[::-1]
    for y, (mean,low,high), color in zip(pos, values, colors):
        ax.errorbar(mean,y,xerr=[[mean-low],[high-mean]],fmt="o",capsize=3.3,color=color,
                    markersize=4.5,lw=1.5,zorder=3)
    ax.axvline(0,color="#75808b",lw=.7)
    ax.set_yticks(pos,names)
    ax.set_ylim(-.6,len(names)-.4)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.grid(axis="x",alpha=.15)


def main():
    human=["Restored − dropped","Full − dropped"]
    contrasts=["restored_minus_dropped","full_minus_dropped"]
    sources=["exp_results.json","exp_locked_results.json"]
    keys=["primary_simultaneous","primary_simultaneous_ci"]
    frames=[json.loads((ROOT/p).read_text())[k] for p,k in zip(sources,keys)]
    twins=json.loads((ROOT/"exp_twins.json").read_text())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42})
    fig,axs=plt.subplots(1,3,figsize=(9.1,2.45),gridspec_kw={"width_ratios":[1,1,.86]})
    palette=["#20697a","#755397"]
    for ax,frame,title in zip(axs,frames,["A   Archived target","B   Locked new target"]):
        ax.set_title(title,loc="left",fontsize=9,weight="bold",pad=10)
        add_points(ax,[frame[name][0] for name in contrasts],human,palette)
        ax.set_xlabel("Brier loss difference")
    left=min(frame[k][0][1] for frame in frames for k in contrasts)
    for ax in axs[:2]:
        ax.set_xlim(left*1.1, -left*.055)
        ax.set_xticks([-.045,-.030,-.015,0.])
    values=np.array(twins["mean_lo_hi"])[:,1]
    ax=axs[2]
    ax.set_title("C   Exact-alignment twins",loc="left",fontsize=9,weight="bold",pad=10)
    add_points(ax,values,["Inert","Strong"],["#637380","#9c5b23"])
    ax.set_xlabel("Projected Brier floor")
    ax.set_xlim(-.008,.10)
    fig.tight_layout(w_pad=1.4)
    for suffix in ("pdf","png"):
        fig.savefig(ROOT/f"exp_readout66.{suffix}",dpi=200,bbox_inches="tight",pad_inches=.025)
    plt.close(fig)


if __name__=="__main__":
    main()
