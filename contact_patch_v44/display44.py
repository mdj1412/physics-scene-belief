"""Render the prespecified projection contrasts and tool-aware forecasts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
stats = json.loads((root / "stats44.json").read_text())
ev = json.loads((root / "forecast_eval44.json").read_text())
colors = {"fold": "#185e87", "ripple": "#a44c23"}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
a, b = axes
a.axhline(0, color="#737373", lw=.8)
for i, shape in enumerate(colors):
    for j, condition in enumerate(("Nominal", "Shifted")):
        mean, lower, upper = np.array(stats["primary"])[i, j] * 1000
        x = i * 2.4 + j * .75
        a.errorbar(x, mean, [[mean-lower], [upper-mean]], fmt="o", ms=8,
                   color=colors[shape], markerfacecolor=colors[shape] if j else "white",
                   capsize=4, linewidth=2)
        a.text(x, lower - .55, f"{mean:+.2f}", ha="center", color=colors[shape], fontsize=9)
a.set(xticks=[0,.75,2.4,3.15], xticklabels=["Nominal","Shifted","Nominal","Shifted"],
      xlim=(-.65,3.8), ylim=(-.7,8.4), ylabel="Projection − original Brier (×1,000)",
      title="A   Effect averaged across four tools")
for p, name in [(0.375,"Fold"),(2.775,"Ripple")]:
    a.text(p,8,name,ha="center",color=colors[name.lower()],weight="bold")
for cell in ev["cells"]:
    if cell["scope"] != "all":
        continue
    changed = np.sign(cell["pred"]) != np.sign(cell["carry"])
    b.scatter(cell["pred"]*1000,cell["observed"]*1000,s=64 if changed else 30,
              c=colors[cell["shape"]],marker="D" if changed else "o",
              edgecolors="black" if changed else "white",lw=.8,alpha=.9,zorder=3)
lims=(-17,23)
b.plot(lims,lims,color="#9c9c9c",lw=.8,ls="--")
b.axvline(0, color="#737373",lw=.8)
b.axhline(0, color="#737373",lw=.8)
from matplotlib.lines import Line2D
handles=[Line2D([],[],marker="o",color="none",markerfacecolor=c,markeredgecolor=c,label=n.title())
         for n,c in colors.items()]
handles.append(Line2D([],[],marker="D",color="none",markerfacecolor="white",markeredgecolor="black",
                      label="Sign changed from v43"))
pts=[c for c in ev["cells"] if c["scope"]=="all"]
extent=[1000*c[k] for c in pts for k in ["pred","observed"]]
lims=(min(extent)-1.5,max(extent)+1.5)
b.set(xlim=lims,ylim=lims,xlabel="Forecast Brier difference (×1,000)",
      ylabel="Held-out Brier difference (×1,000)", title="B   Forecasts frozen before target labels")
b.legend(handles=handles,loc="upper left",fontsize=8,frameon=False)
for ax in axes:
    ax.spines[["right","top"]].set_visible(False)
    ax.grid(axis="y",alpha=.14)
fig.savefig(root / "contact_patch_v44.png",dpi=180)
