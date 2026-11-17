"""Crossover on one GPU; one selected likelihood per paid observation, charged frame transfer."""
import json
import time
import numpy as np
from runner46 import ROOT,PROJECT,SHAPES,old,infer,r,v38,torch,timing45
from timing_repeat45 import snapshot

METHODS=("always_original","legacy_x","refit_x","x_normal")

def measure(engine,axes,block):
    from camera46 import render_clip,infer_x,infer_normal_variance,source_gate,gate46,load_lookup
    lookup=load_lookup("cuda")
    output=ROOT/("latency46" if block=="timing" else "latency46_repeat")
    output.mkdir(exist_ok=True)
    snapshots=[]
    for panel in range(12):
        f=(panel//2)%4
        cell=(SHAPES[panel%2],old.world.TARGET_SURFACES[(f//2)%2],old.world.TARGET_LAWS[f%2],
              old.world.VARIANTS[(panel//2)%3],old.world.HEIGHTS[(panel//2)%2])
        theta,xx,sensors,_=old.observations(panel,cell)
        one=[{a:row[a][:,:1] for a in (r.Q,r.P)} for row in sensors]
        h=round(1000*cell[-1])
        templates=[infer.make_template(engine,cell[-1]),infer.make_template(engine,cell[-1],axes[str(h)]["axis"])]
        with np.load(old.path_for(panel,cell)) as a:
            reference=a["pred"][:,0,:,:,0]
            selected=a["selected"]
        rng=np.random.default_rng(460001337+panel)
        offset=rng.permutation(np.resize(np.array([-.5,0.,.5]),512))
        clip=render_clip(theta.reshape(512,6),cell[0],cell[3],h,offset,460044113+panel,device="cpu").contiguous()
        assert clip.dtype==torch.uint8 and tuple(clip.shape)==(512,3,32,96)
        rng=np.random.default_rng(460020305+panel+(block=="repeat")*316)
        wall=np.zeros((4,len(METHODS)));feature=np.zeros_like(wall);order=np.zeros_like(wall,int)
        rates=np.zeros(len(METHODS));err=0.
        before=snapshot()
        for rep in range(-1,len(wall)):
            seq=rng.permutation(len(METHODS))
            for m in seq:
                torch.cuda.synchronize()
                start=time.perf_counter_ns()
                if m==0:
                    route=np.zeros((4,128),bool)
                else:
                    frames=clip.to("cuda")
                    x=infer_x(frames)
                    if m==1:
                        gate=source_gate(x)
                    elif m==2:
                        gate=gate46(x,None,lookup)
                    else:
                        v=infer_normal_variance(frames,x)
                        gate=gate46(x,v,lookup)
                    route=gate.cpu().numpy().reshape(4,128)
                torch.cuda.synchronize()
                ready=time.perf_counter_ns()
                act=infer.actions43(xx,engine)
                paid=infer.paid_sensors(one,act)
                answer=timing45.decode_mixed(engine,templates,xx,act,paid,route)
                torch.cuda.synchronize()
                end=time.perf_counter_ns()
                np.testing.assert_array_equal(act,selected)
                expected=np.where(route,reference[1],reference[0])
                np.testing.assert_allclose(answer,expected,atol=2e-9,rtol=0.)
                err=max(err,float(np.max(abs(expected-answer))))
                rates[m]=route.mean()
                if rep>=0:
                    wall[rep,m]=(end-start)/1e6
                    feature[rep,m]=(ready-start)/1e6
            if rep>=0:order[rep]=seq
        snapshots.append(dict(panel=panel,before=before,after=snapshot()))
        v38.atomic_npz(output/f"panel_{panel:02d}.npz",warm_ms=wall,feature_ms=feature,order=order,methods=METHODS,
                       rates=rates,err=err,image_bytes=clip.numel(),cell=[str(s) for s in cell])
        v38.atomic_json(output/"resources.json",snapshots)
        print("timing",block,panel,flush=True)
