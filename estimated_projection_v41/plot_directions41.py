"""Render four measured spatial deformation modes, with orientation aligned only for display."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
records = json.loads((HERE / "direction_audit41.json").read_text())["cells"]
fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4), sharex=True, sharey=True, layout="constrained")
for ax, (name, group) in zip(axes.flat, records.items()):
    family, height = name.split("_")
    oracle = np.array(group["oracle"])
    estimated = np.array([row["direction"] for row in group["panel"]])
    estimated *= np.sign(estimated @ oracle)[:, None]
    offset = np.arange(-6, 7, 2)
    ax.axvline(0, color="#a0a7ae", lw=.8, ls=":")
    ax.axhline(0, color="#a0a7ae", lw=.8)
    for panel in estimated:
        ax.plot(offset, panel, color="#37939a", lw=.7, alpha=.28)
    estimated_line, = ax.plot(offset, estimated.mean(0), "-o", color="#07717b", lw=2,
                              ms=3.5, label="Paired estimate (12-panel mean)")
    oracle_line, = ax.plot(offset, oracle, "--s", color="#ad4b14", lw=1.7, ms=3.5, label="Oracle axis")
    ax.set_title(f"{family.capitalize()}   ·   {height} mm")
    ax.set_xticks(offset)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-1., 1.)
for ax in axes[-1]:
    ax.set_xlabel("Offset from query contact height (mm)")
for ax in axes[:,0]:
    ax.set_ylabel("Normalized component")
fig.legend(handles=(estimated_line,oracle_line), loc="outside lower center", ncol=2, frameon=False)
fig.savefig(HERE / "geometry_directions41.png", dpi=180)
