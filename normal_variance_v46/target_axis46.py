"""Post-hoc diagnostic of known target-family hollow-to-bulge contrast at a nominal height."""
from pathlib import Path
import argparse
import json
import numpy as np
from runner46 import ROOT,PROJECT,SHAPES,old,infer,r,v38,torch,configure
from geometry46 import shape_profile
import os

def directions():
    source=json.loads((PROJECT/"image_pressure_v45/source_axis45.json").read_text())
    result={}
    for shape in SHAPES:
        result[shape]={}
        for h in (19,31):
            y=old.world.w.view_heights(h/1000.)
            y=np.asarray(torch.as_tensor(y).detach().cpu())*1000
            sigma=np.asarray(old.world.w.sigmas()[r.Q].detach().cpu())
            u=(shape_profile(y,shape,"bulge")[0]-shape_profile(y,shape,"hollow")[0])*.001/sigma
            v=u/np.linalg.norm(u)
            result[shape][str(h)]=dict(unit=v.tolist(),contrast_whitened_norm=float(np.linalg.norm(u)),
                abs_cosine_source=float(abs(v@np.array(source[str(h)]["axis"]))))
    return result

def main():
    if not os.getenv("SLURM_JOB_ID") or torch.cuda.device_count()!=1:raise RuntimeError("Use an allocated GPU")
    torch.set_num_threads(1);configure(True)
    result=directions()
    output=ROOT/"diag_axis46"
    output.mkdir(exist_ok=True)
    v38.atomic_json(output/"axes.json",result)
    paths=0;maxerr=0.
    with torch.inference_mode():
        engine=infer.make_bank()
        templates={(shape,h):infer.make_template(engine,h/1000.,result[shape][str(h)]["unit"])
                   for shape in SHAPES for h in (19,31)}
        for n in range(12):
            names=sorted((old.OUT/f"panel_{n:02d}").glob("*.npz"))
            assert len(names)==48
            for p in names:
                dest=output/p.parent.name/p.name
                dest.parent.mkdir(exist_ok=True,parents=True)
                shape=p.stem.split("_")[0];h=int(p.stem.split("_")[-1][1:])
                with np.load(p) as a:
                    xx=a["xx"];acts=a["selected"];orig=a["pred"][0,0];q=a["query_paid"];probe=a["probe_paid"]
                obs=[{r.Q:q[e],r.P:probe[e]} for e in range(len(xx))]
                pred=infer.decode43(engine,templates[shape,h],xx,acts,obs)
                assert pred.shape==(4,128,4) and np.isfinite(pred).all()
                assert pred.min()>-1e-12 and pred.max()<1+1e-12
                maxerr=max(maxerr,float(abs(pred[acts==r.P]-orig[acts==r.P]).max()))
                v38.atomic_npz(dest,pred=pred)
                paths+=1
            print("target axis",n,flush=True)
    assert paths==576 and maxerr<1e-12
    v38.atomic_json(output/"complete.json",dict(archives=paths,probabilities=paths*4*128*4,max_probe_error=maxerr,
          diagnostic="known analytic target paired-variant contrast; families and nominal h known; fixed original paid sensor & frozen bank"))
if __name__=="__main__":main()
