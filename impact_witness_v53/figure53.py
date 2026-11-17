"""Separate policy effects and fourth-observation effects in units of 1e-4 Brier."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root/'score53.json').read_text())
rows = [
    ("Same observations: 3-action selector", data["primary"]["impact_selector"]),
    ("Same observations: 3-action vote", data["primary"]["impact_vote_selector"]),
    ("Same observations: post-only vote", data["secondary"]["impact_post_vote"]),
    ("Same decoder: four reference pairs", data["secondary"]["reference4_joint"]),
    ("Frozen v52: selector, four references", data["secondary"]["v52_selector"]),
    ("Same decoder: impact sign hidden", data["secondary"]["impact_joint_masked"]),
]
fig, axes = plt.subplots(1,2, figsize=(10.4,3.65),sharey=True)
for ax, field, title in zip(axes, ("mean","p90"), ("Mean held-out Brier loss","Session Brier loss, 90th percentile")):
    for j,(_,result) in enumerate(rows):
        mu, lo, hi=np.array(result[field])*1e4
        ax.errorbar(mu,j,xerr=[[mu-lo],[hi-mu]],color="#00777c" if j<2 else "#646878",
                    marker="o", ms=5,capsize=3)
    ax.axvline(0, color="#434343",lw=.8)
    ax.set_title(title)
    ax.set_xlabel("Loss difference (×10⁻⁴)")
    ax.grid(axis="y",alpha=.15)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_yticks(range(len(rows)),[r[0] for r in rows])
axes[0].invert_yaxis()
fig.tight_layout(w_pad=2)
fig.savefig(root/"comparisons53.png",dpi=170,bbox_inches="tight")
