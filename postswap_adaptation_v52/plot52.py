"""Plot mean and session-tail scores from the saved deployment evaluation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

root = Path(__file__).resolve().parent
scores = json.loads((root / "score52.json").read_text())
styles = [
    ("selector", "Likelihood, 3 actions", "#006b70", "-", "o"),
    ("vote_selector", "Sign vote, 3 actions", "#ae5d1e", "--", "s"),
    ("continuous_joint", "Continuous likelihood", "#7358a5", "-.", "^"),
    ("unsigned", "Unsigned", "#55585b", ":", None),
]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
for ax, values, title in zip(axes, [scores["mean"][0], scores["absolute_p90"]],
                             ["Mean held-out Brier loss", "Session Brier loss, 90th percentile"]):
    for name, label, color, line, marker in styles:
        column = scores["methods"].index(name)
        ax.plot(scores["K"], [row[column] for row in values], label=label,
                linestyle=line, marker=marker, color=color, markersize=4)
    ax.set(title=title, xlabel="Post-swap calibration pairs", xticks=scores["K"])
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, .14, 1, 1), w_pad=2)
fig.savefig(root / "budget52.png", dpi=170, bbox_inches="tight")
