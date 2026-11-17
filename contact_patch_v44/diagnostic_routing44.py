"""Exploratory offline diagnostic: value of applying the frozen sign by design cell."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

root = Path(__file__).resolve().parent
records = [r for r in json.loads((root / "forecast44.json").read_text())["rows"] if r["scope"] == "all"]
d = np.array(json.loads((root / "stats44.json").read_text())["fine_panel_d"])
shapes, heights, radii, spans, offsets = ("fold","ripple"), (19,31), (.5,2.), (0.,2.), (-.5,0.,.5)
routes = {}
for name in ("pred", "carry"):
    route = np.full(d.shape[1:], np.nan)
    for r in records:
        idx = (shapes.index(r["shape"]), heights.index(r["height_mm"]), radii.index(r["r_mm"]),
               spans.index(r["span_mm"]), offsets.index(r["offset_mm"]))
        assert np.isnan(route[idx])
        route[idx] = r[name] < 0
    assert np.isfinite(route).all()
    routes[name] = route
def interval(x):
    panel = x.mean(axis=(1,2,3,4,5))
    m = panel.mean()
    w = t.ppf(1-.025/2,11) * panel.std(ddof=1) / np.sqrt(12)
    return [m,m-w,m+w]
out = dict(equal_weight_conditions=48, projected_cells={k:int(a.sum()) for k,a in routes.items()},
           versus_original=interval(d*routes["pred"]), versus_carry=interval(d*(routes["pred"]-routes["carry"])),
           estimand="Post-hoc retrospective use of a pre-label diagnostic (known true signed offset). Two paired comparisons.")
(root / "diagnostic_routing44.json").write_text(json.dumps(out,indent=2)+"\n")
