"""Render readout probabilities and product-balanced errors from the saved results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
results = json.loads((folder / "summary.json").read_text())
reference = json.loads((folder.parent / "review" / "finite_reference.json").read_text())["rubbers"]
budgets = list(map(str, results["budgets"]))
x = list(range(len(budgets)))
fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
for rubber, label, color in [("grass_dtecs", "Grass", "#146953"), ("abs_ii_soft", "ABS", "#a34c25")]:
    for field, name, style in [("strict_majority", "flip", "-"), ("binomial_one_sided_05", "evidence", "--")]:
        vals = [100 * reference[rubber]["budgets"][k][field]["p_flip"] for k in budgets]
        axes[0].plot(x, vals, style, color=color, marker=".", label=f"{label} {name}")
axes[0].set(title="Gate reversal by budget", ylabel="Probability on calibration pool (%)", ylim=(-3, 105))
axes[0].legend(loc="center right", frameon=False, fontsize=8)
methods = [
    ("gate_only", "Gate only", "#797677", "--"), ("gate_expert", "Gate + experts", "#13795c", "-"),
    ("inclined_rigid", "Inclined rigid", "#af572b", ":"),
    ("single_affine_eb", "Single affine (EB)", "#425e9e", "-."),
    ("gated_expert_eb", "Gate + experts (EB)", "#8866a2", "--"),
    ("elastic_response", "Elastic impulse", "#161e2a", "-")]
for index, axis in enumerate(axes[1:]):
    for method, label, color, style in methods:
        axis.plot(x, [results["overall"][method][k]["macro"][index] for k in budgets], style,
                  marker=".", color=color, label=label, linewidth=1.8)
for axis, title, units in zip(axes[1:], ["Held-out velocity", "Held-out spin"], ["Vector error (m/s)", "MAE (rad/s)"]):
    axis.set(title=title, ylabel=units)
for axis in axes:
    axis.set(xticks=x, xticklabels=budgets, xlabel="Calibration impacts")
    axis.spines[["right","top"]].set_visible(False)
    axis.grid(axis="y", alpha=.2)
handles, names = axes[1].get_legend_handles_labels()
fig.legend(handles, names, loc="lower center", ncol=6, frameon=False, fontsize=9)
fig.tight_layout(rect=(0,.10,1,1))
fig.savefig(folder / "calibration_curves.png", dpi=170)
