"""Remove shared noise and baseline geometry in paired 7-point paid profiles, no outcomes."""
import itertools
import json
from pathlib import Path
import numpy as np
from geometry46 import shape_profile

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/"image_pressure_v45"
def main():
    axes=json.loads((SRC/"source_axis45.json").read_text())
    rows=[]
    for which,folder,families in (("source",SRC,("fold","ripple")),("target",ROOT,("terrace","dimple"))):
        for family,h,(va,vb) in itertools.product(families,(19,31),
                                                 (("hollow","bulge"),("base","bulge"),("hollow","base"))):
            vals=[]
            axis=np.asarray(axes[str(h)]["axis"])
            for panel in range(12):
                stem=folder/"results"/f"panel_{panel:02d}"/f"{family}_inlay_rational"
                with np.load(f"{stem}_{va}_h{h}.npz") as a,np.load(f"{stem}_{vb}_h{h}.npz") as b:
                    np.testing.assert_array_equal(a["theta"],b["theta"])
                    # Same noise stream for both views, restrictions enforce two purchases.
                    diff=(a["query_paid"][:,:,0]-b["query_paid"][:,:,0]).reshape(-1,7)
                diff=diff[np.isfinite(diff).all(1)]
                length=np.linalg.norm(diff,axis=1)
                vals.append(np.abs(diff @ axis)/length)
                if which=="target":
                    x=h+np.arange(-3,4)*2.
                    ideal=(shape_profile(x,family,va)[0]-shape_profile(x,family,vb)[0])*.001
                    np.testing.assert_allclose(diff,ideal+np.zeros_like(diff),atol=1e-12)
            v=np.concatenate(vals)
            rows.append(dict(domain=which,family=family,h_mm=h,pair=[va,vb],
                     paired_paid_profiles=len(v),abs_cos=float(v.mean()),fraction_signal_energy_removed=float((v*v).mean())))
    (ROOT/"axis_transfer46.json").write_text(json.dumps(rows,indent=2)+"\n")
    print([r for r in rows if r["pair"]==["hollow","bulge"]])

if __name__=="__main__":main()
