"""Estimate six source-only sign bins for the source-calibrated projection axis."""
import hashlib
import json
from pathlib import Path
import numpy as np

from runner45 import ROOT, old, infer, r, torch
import physics_check43

def main():
    if torch.cuda.is_available():
        raise RuntimeError("Run the source-only diagnostic on allocated CPU with CUDA hidden")
    torch.set_num_threads(1)
    past = ROOT.parent/"contact_patch_v44/results"
    axes=json.loads((ROOT/"source_axis45.json").read_text())
    risk=np.zeros((12,2,3))
    replay_max = 0.
    engine = infer.make_bank()
    templates = {h:infer.make_template(engine,h/1000.,v["axis"]) for h,v in ((int(k),v) for k,v in axes.items())}
    originals = {h:infer.make_template(engine,h/1000.) for h in templates}
    with torch.inference_mode():
        for panel in range(12):
            cache = {}
            for shp, surf, law, var, h0 in old.CELLS:
                h=round(h0*1000)
                with np.load(past/f"panel_{panel:02d}"/f"{shp}_{surf}_{law}_{var}_h{h}.npz") as a:
                    xx = a["xx"][:,:8]; act=a["selected"][:,:8]
                    q,p = a["query_paid"][:,:8], a["probe_paid"][:,:8]
                    theta,sg,regs=a["theta"][:,:8],a["signs"],a["regimes"]
                    p0=a["pred"][0,0,:,:8]
                key = (h,hashlib.sha256(xx.tobytes()+act.tobytes()+q.tobytes()+p.tobytes()).digest())
                if key not in cache:
                    sens=[{r.Q:vq,r.P:vp} for vq,vp in zip(q,p)]
                    if not cache:
                        replay=infer.decode43(engine,originals[h],xx,act,sens)
                        np.testing.assert_allclose(p0,replay,atol=2e-9,rtol=0)
                        replay_max=max(replay_max,float(np.abs(p0-replay).max()))
                    cache[key] = infer.decode43(engine,templates[h],xx,act,sens)
                pg=cache[key]
                for j,delta in enumerate((-.5,0.,.5)):
                    y=np.stack([physics_check43.score(te,shp,var,surf,law,s,reg,h+delta)
                                for te,s,reg in zip(theta,sg,regs)])[:,:,None]
                    risk[panel,(19,31).index(h),j]+=((pg-y)**2-(p0-y)**2).mean()/24
    d=risk.mean(0).ravel()
    output=dict(centers_mm=[18.5,19,19.5,30.5,31,31.5],D=d.tolist(),project=(d<0).tolist(),
                source="v44 historical paid predictions, generic source axis, legacy v43 exact centered labels",
                nominal_heights_mm=[19,31],first_objects_per_environment=8,environments=4,panels=12,
                source_brier_diff_by_panel=risk.tolist(),source_original_max_replay_error=replay_max)
    (ROOT/"matched_source45.json").write_text(json.dumps(output,indent=2)+"\n")

if __name__=="__main__":
    main()
