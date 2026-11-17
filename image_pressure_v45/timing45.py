"""Warm end-to-end same-device crossover for three decoder routing policies."""
import json
import time
import numpy as np
from runner45 import ROOT, old, infer, r, v38, torch

METHODS=("always_original","oracle_x","visual_x","matched_oracle_x","matched_visual_x")

def decode_mixed(engine,templates,xx,act,paid,project):
    """Each purchased observation goes to a single decoder likelihood."""
    out = np.empty(xx.shape[:2],float)
    bank = engine.inner
    for e in range(len(xx)):
        for sensor,di in ((r.P,0),(r.Q,0),(r.Q,1)):
            chosen = act[e]==sensor
            if sensor==r.Q:
                chosen = chosen & (project[e]==bool(di))
            idx=np.flatnonzero(chosen)
            if not len(idx):
                continue
            white,norm,label = templates[di]
            prior = bank.fit(v38.t(xx[e,idx]))[:,None,:]
            y = v38.t(paid[e][sensor][idx])
            if sensor==r.Q:
                ll=(y/bank.sigma[sensor])@white.T-norm
                post=torch.softmax(prior.clamp_min(1e-300).log()+ll,-1)
            else:
                post=bank.update(prior,sensor,y)
            out[e,idx]=(post@label)[:,0].cpu().numpy()
    return out

def run(panel,engine,axes):
    from camera45 import render_clip,infer_x,source_gate
    source=json.loads((ROOT/"matched_source45.json").read_text())
    centers=torch.tensor(source["centers_mm"],device="cuda")
    matched=torch.tensor(source["project"],device="cuda")
    f=(panel//2)%4
    cell=(old.world.TARGET_SHAPES[panel%2],old.world.TARGET_SURFACES[(f//2)%2],
          old.world.TARGET_LAWS[f%2],old.world.VARIANTS[(panel//2)%3],old.world.HEIGHTS[(panel//2)%2])
    t,xx,sensors,_=old.observations(panel,cell)
    one=[{a:row[a][:,:1] for a in (r.Q,r.P)} for row in sensors]
    h=round(1000*cell[-1])
    items=[infer.make_template(engine,h/1000),infer.make_template(engine,h/1000,axes[str(h)]["axis"])]
    with np.load(old.path_for(panel,cell)) as a:
        reference=a["pred"][:2,0,:,:,0]
        selected=a["selected"]
    rng=np.random.default_rng(450003357+panel)
    offset=rng.permutation(np.resize(np.array([-.5,0.,.5]),512))
    clip=render_clip(t.reshape(512,6),cell[0],cell[3],h,offset,450080132+panel,device="cpu").contiguous()
    if clip.dtype!=torch.uint8 or clip.shape[0]!=512:
        raise ValueError("Expected a raw uint8 image sequence for every episode")
    x_gt=torch.as_tensor(h+offset,dtype=torch.float64)
    seeds=np.zeros((3,len(METHODS)),int)
    ms=np.zeros_like(seeds,float)
    locate_ms=np.zeros_like(ms)
    error=0.
    frequencies=np.zeros(len(METHODS))
    for rep in range(-1,3):
        ids=rng.permutation(len(METHODS))
        if rep>=0:seeds[rep]=ids
        for m in ids:
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            if m==0:
                route=np.zeros((4,128),bool)
            else:
                x=x_gt.to("cuda") if m in (1,3) else infer_x(clip.to("cuda"))
                gate=source_gate(x) if m < 3 else matched[(x[...,None]-centers).abs().argmin(dim=-1)]
                route=gate.cpu().numpy().reshape(4,128)
            torch.cuda.synchronize()
            loc_end=time.perf_counter_ns()
            act=infer.actions43(xx,engine)
            paid=infer.paid_sensors(one,act)
            answer=decode_mixed(engine,items,xx,act,paid,route)
            torch.cuda.synchronize()
            end=time.perf_counter_ns()
            np.testing.assert_array_equal(act,selected)
            expected=np.where(route,reference[1],reference[0])
            np.testing.assert_allclose(answer,expected,atol=2e-9,rtol=0)
            error=max(error,float(abs(answer-expected).max()))
            frequencies[m]=route.mean()
            if rep>=0:
                ms[rep,m]=(end-start)*1e-6
                locate_ms[rep,m]=(loc_end-start)*1e-6
    path=old.OUT/"timing"/f"panel_{panel:02d}.npz"
    path.parent.mkdir(parents=True,exist_ok=True)
    v38.atomic_npz(path,warm_ms=ms,location_ms=locate_ms,order=seeds,methods=METHODS,max_error=error,
                   projection_share=frequencies,timing_cell=np.asarray([str(v) for v in cell]),
                   image_shape=np.array(clip.shape),image_bytes=clip.numel())
