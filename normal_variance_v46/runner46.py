"""Compute held-out v46 probabilities from factual and one paid observation."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/"image_pressure_v45"))
import timing45
from runner45 import old, infer, r, v38, torch
DECODERS=("original","source_axis")
SHAPES=("terrace","dimple")

def configure(with_geometry):
    old.OUT=ROOT/"results"
    old.BASE,old.SUPPORT_BASE=460000000,600
    if with_geometry:
        import geometry46
        assert tuple(geometry46.SHAPES)==SHAPES
        geometry46.install_world43(old.world)
    else:
        old.world.TARGET_SHAPES=SHAPES
    old.CELLS=tuple(itertools.product(SHAPES,old.world.TARGET_SURFACES,old.world.TARGET_LAWS,old.world.VARIANTS,old.world.HEIGHTS))

def collect(panel,cell,engine,templates,cache,axes,fingerprint):
    path=old.path_for(panel,cell)
    if r.old.already_complete(path,fingerprint):
        return
    theta,xx,sensors,_=old.observations(panel,cell)
    key=hashlib.sha256(xx.tobytes()).digest()
    if key not in cache:
        cache[key]=infer.actions43(xx,engine)
    act=cache[key]
    paid=infer.paid_sensors(sensors,act)
    h=cell[-1]
    if h not in templates:
        templates[h]=[infer.make_template(engine,h,axis) for axis in (None,axes[str(round(1000*h))]["axis"])]
    pred=np.stack([infer.decode43(engine,tmp,xx,act,paid) for tmp in templates[h]])[:,None]
    assert pred.shape==(2,1,4,128,4) and np.isfinite(pred).all()
    assert pred.min()>=-1e-12 and pred.max()<=1+1e-12
    np.testing.assert_allclose(pred[0,0][act==r.P],pred[1,0][act==r.P],atol=1e-12,rtol=0)
    path.parent.mkdir(parents=True,exist_ok=True)
    v38.atomic_npz(path,theta=theta,xx=xx,selected=act,pred=pred,decoders=DECODERS,policies=["zero"],
                   query_paid=np.stack([m[r.Q] for m in paid]),probe_paid=np.stack([m[r.P] for m in paid]))
    v38.atomic_json(path.with_suffix(".json"),dict(sha256=v38.digest(path),fingerprint=fingerprint))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("population","even","odd","timing","repeat"));mode=ap.parse_args().mode
    if not os.getenv("SLURM_JOB_ID"):raise RuntimeError("Requires an allocated compute node")
    torch.set_num_threads(1)
    configure(mode!="population")
    if mode=="population":
        for panel in range(12):old.manifest(panel)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("Expected one allocated GPU")
    hashes=json.loads((PROJECT/"blind_decoder_v43/results/run_even.json").read_text())["hashes"]
    for name in ("gpu_freeze46.sha256",):
        for row in (ROOT/name).read_text().splitlines():
            digest,rel=row.split(maxsplit=1)
            if v38.digest(ROOT/rel)!=digest:raise ValueError(rel)
            hashes[ROOT.name+"/"+rel]=digest
    for path,digest in hashes.items():
        if v38.digest(PROJECT/path)!=digest:raise ValueError(path)
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    meta=dict(mode=mode,job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
              torch=str(torch.__version__),fingerprint=fingerprint,hashes=hashes,decoders=DECODERS,
              seed=old.BASE,support_start=old.SUPPORT_BASE,unseen_families=SHAPES)
    old.OUT.mkdir(exist_ok=True)
    v38.atomic_json(old.OUT/f"run_{mode}.json",meta)
    axes=json.loads((PROJECT/"image_pressure_v45/source_axis45.json").read_text())
    with torch.inference_mode():
        engine=infer.make_bank()
        templates={}
        if mode in ("timing","repeat"):
            import timing46
            timing46.measure(engine,axes,mode)
        else:
            for panel in range(int(mode=="odd"),12,2):
                cache={}
                order=np.random.default_rng(old.BASE+100000*panel+707).permutation(len(old.CELLS))
                for k in order:collect(panel,old.CELLS[k],engine,templates,cache,axes,fingerprint)
                print("finished",panel,flush=True)
    v38.atomic_json(old.OUT/f"complete_{mode}.json",meta)

if __name__=="__main__":
    main()
