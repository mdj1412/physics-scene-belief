"""Draw the two v66 comparisons from committed summaries on a Slurm CPU."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    if not os.getenv("SLURM_JOB_ID") or os.getenv("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use a Slurm compute step with CUDA disabled.")
    soft_source = read("soft_choice.json")
    soft_target = read("soft_transfer.json")
    readout = read("exp_results.json")

    grid = np.asarray(soft_target["grid"])
    source = np.asarray(soft_source["source_mean"])
    target = np.asarray(soft_target["mean"])
    assert len(grid) == len(source) == len(target) and grid[0] == 0 and grid[-1] == 1
    alpha = soft_source["alpha"]
    assert alpha == soft_target["alpha"] == grid[np.argmin(source)]
    assert readout["metrics"][0] == "all purchased actions"
    assert len(readout["support_ids"]) == 12 and len(readout["source_seeds"]) == 3
    assert soft_target["source_families"] == ["fold", "ripple"]
    assert soft_target["target_families"] == ["terrace", "dimple"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "ytick.labelsize": 8,
        "xtick.labelsize": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#52626b",
        "axes.linewidth": .7,
        "xtick.color": "#263842",
        "ytick.color": "#263842",
        "axes.labelcolor": "#1d2d35",
        "text.color": "#192a33",
    })
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(6.45, 2.60), gridspec_kw={"width_ratios": [1.2, .94]})
    fig.subplots_adjust(left=.105, right=.977, bottom=.23, top=.84, wspace=.47)
    for ax in (left, right):
        ax.grid(axis="y", color="#dbe1e4", alpha=.85, linewidth=.6)
        ax.axhline(0, color="#5a6b73", lw=.85, zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=3)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    left.set_title("a   Frozen likelihood", loc="left", fontweight="semibold", fontsize=10, pad=7)
    for arr, label, color, marker in (
        (source, "Source: fold, ripple", "#175872", "o"),
        (target, "Target: terrace, dimple", "#ad502c", "s"),
    ):
        left.plot(grid, (arr - arr[0]) * 1000, marker=marker, color=color,
                  markersize=4.2, lw=1.6, label=label, zorder=3)
    left.axvline(alpha, ls=(0, (2, 2)), lw=1, color="#657078", zorder=1,
                 label=fr"Source choice: $\alpha$ = {alpha:g}")
    left.set_xlim(-.055, 1.055)
    left.set_xticks(grid, [f"{x:g}" for x in grid])
    left.set_xlabel(r"Suppression strength $\alpha$")
    left.set_ylabel(r"$10^3$ × Brier change vs. full")
    left.legend(loc="best", frameon=True, fancybox=False, framealpha=.95,
                edgecolor="none", fontsize=7.0, handlelength=1.5,
                borderpad=.23, labelspacing=.35)

    right.set_title("b   Readouts fit on source", loc="left", fontweight="semibold", fontsize=10, pad=7)
    keys = ("full_minus_dropped", "restored_minus_dropped")
    for index, key in enumerate(keys):
        # Two simultaneous contrasts, complete-policy metric (paid probes included).
        center, lower, upper = np.asarray(readout["primary_simultaneous"][key][0]) * 1000
        assert lower <= center <= upper
        right.errorbar(index, center, yerr=[[center-lower], [upper-center]],
                       color="#ad502c", fmt="s", markersize=5, lw=1.5,
                       capsize=4, capthick=1.2, zorder=3)
    right.set_xlim(-.48, 1.48)
    right.set_xticks([0, 1], ["Full", "Restored"])
    right.set_xlabel("Profile coordinates in readout")
    right.set_ylabel(r"$10^3$ × Brier change vs. dropped")
    right.margins(y=.17)
    fig.savefig(ROOT / "figure66.pdf", metadata={
        "Title": "Geometry suppression and refitted forecasts on transferred contours",
        "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
