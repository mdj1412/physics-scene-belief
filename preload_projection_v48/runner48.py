"""Generate label-blind GPU forecasts and time the single-decoder online paths."""
import argparse
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np

from population48 import FAMILIES, HEIGHTS, ROOT, PROJECT, old, infer, r, v38, torch
sys.path.insert(0, str(PROJECT / "factorial_projection_v47"))
from runner47 import apparatus  # exact factual and query/probe readout inherited from v47
import timing45
import world48
import sensor48
import route48

METHODS = ("always_original", "fixed", "preload", "none")
MODES = ("fixed", "preload", "none", "unsigned")
VARIANTS = ("hollow", "bulge")
ALIGN = ("low", "high")
STATES = (0, -1, 1)
AXES = json.loads((PROJECT / "image_pressure_v45/source_axis45.json").read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world(panel):
    with np.load(ROOT / "worlds" / f"panel_{panel:02d}.npz") as storage:
        return {key: storage[key] for key in storage}


def add_optical(sensors, family, height, a, variant):
    delta_m = .001 * world48.sensor_delta_mm(family, height, a, variant)
    assert np.shape(delta_m) == (7,)
    return [{r.Q: s[r.Q] + delta_m, r.P: s[r.P]} for s in sensors]


def all_readings(pop):
    readings = np.empty((4, 2, 3, 2, 4, 128, 4), np.float32)
    for f, family in enumerate(FAMILIES):
        for h, query_height in enumerate(HEIGHTS):
            for s, state in enumerate(STATES):
                for z, station in enumerate(HEIGHTS):
                    measured = sensor48.readings(pop, family, query_height, state, station)
                    assert measured.shape == (4, 128, 4)
                    readings[f, h, s, z] = measured
    assert np.isfinite(readings).all()
    return readings


def gates(q, measurement, h, mode, policy):
    return route48.decision(q, measurement, h, mode, policy).cpu().numpy()


def one_panel(panel, engine, templates, policy, fixed_i):
    pop = world(panel)
    measured = all_readings(pop)
    o = np.empty((4, 2, 2, 2, 4, 128, 4), np.float64)
    p = np.empty_like(o)
    routes = {m: np.empty((4, 2, 2, 2, 3, 4, 128, 4), bool) for m in MODES}
    chosen = np.empty((4, 2, 2, 4, 128), np.int8)
    qpaid = np.empty(o.shape+(7,), np.float32)
    cache = {}
    for f, family in enumerate(FAMILIES):
        for h, height in enumerate(HEIGHTS):
            items = templates[h]
            for v, variant in enumerate(VARIANTS):
                xx, sensors = apparatus(pop, family, height, variant)
                key = hashlib.sha256(xx.tobytes()).digest()
                if key not in cache:
                    cache[key] = infer.actions43(xx, engine)
                act = cache[key]
                chosen[f,h,v] = act
                for a, alignment in enumerate(ALIGN):
                    paid = infer.paid_sensors(add_optical(sensors, family, height, alignment, variant), act)
                    for k, buf in enumerate((o,p)):
                        buf[f,a,h,v] = infer.decode43(engine, items[k], xx, act, paid)
                    q = np.stack([row[r.Q] for row in paid]).astype(np.float32)
                    qpaid[f,a,h,v] = q
                    for mode in MODES:
                        z = fixed_i if mode == "fixed" else h
                        for s in range(len(STATES)):
                            # Neither station index, latent state nor true alignment enters the route.
                            reading = np.zeros_like(measured[f,h,s,z]) if mode=="none" else measured[f,h,s,z]
                            routes[mode][f,a,h,v,s] = gates(q, reading, height, mode, policy)
                if h == fixed_i:
                    np.testing.assert_array_equal(routes["fixed"][f,:,h,v], routes["preload"][f,:,h,v])
        print("forecast",panel,family,flush=True)
    mask = np.broadcast_to(chosen[:,None,:,:,:,:,None] == r.Q,o.shape)
    assert np.all(np.isfinite(qpaid) == mask[...,None])
    np.testing.assert_allclose(o[~mask],p[~mask],atol=1.e-12,rtol=0)
    assert np.isfinite([o,p]).all() and min(o.min(),p.min())>=-1.e-12 and max(o.max(),p.max())<=1+1.e-12
    for g in routes.values():
        assert not np.any(g & ~mask[:,:,:,:,None])
    assert np.all(routes["none"]==routes["none"][:,:,:,:,:1])
    dest=ROOT/"forecast"/f"panel_{panel:02d}.npz"
    dest.parent.mkdir(exist_ok=True)
    v38.atomic_npz(dest,p_original=o,p_projected=p,**{"gate_"+k:v for k,v in routes.items()},
                   readings_mm=measured,query_paid=qpaid,selected=chosen,families=FAMILIES,alignments=ALIGN,
                   variants=VARIANTS,heights_mm=HEIGHTS,stations_mm=HEIGHTS,states=STATES,
                   fixed_height_mm=HEIGHTS[fixed_i],panel=panel,query_id=r.Q,probe_id=r.P)
    v38.atomic_json(dest.with_suffix(".json"),dict(sha256=digest(dest),n_action_groups=len(cache)))


def time_panel(panel,engine,templates,policy,fixed_i):
    from timing_repeat45 import snapshot
    pop=world(panel)
    f=panel%4
    h=(panel//4+panel//2)%2
    a=(panel//4+panel)%2
    v=(panel//2+panel//4)%2
    height=HEIGHTS[h]
    xx,sensors=apparatus(pop,FAMILIES[f],height,VARIANTS[v])
    one=[{k:row[k][:,:1] for k in (r.Q,r.P)} for row in add_optical(sensors,FAMILIES[f],height,ALIGN[a],VARIANTS[v])]
    rng=np.random.default_rng(480305+panel)
    state=rng.permutation(np.repeat(np.arange(3),[256,128,128])).reshape(4,128)
    env,obj=np.indices((4,128))
    with np.load(ROOT/"forecast"/f"panel_{panel:02d}.npz") as arc:
        op=arc["p_original"][f,a,h,v,...,0]
        pp=arc["p_projected"][f,a,h,v,...,0]
        chosen=arc["selected"][f,h,v]
        expected={}
        raw={}
        for mode in METHODS[1:]:
            z=fixed_i if mode=="fixed" else h
            expected[mode]=arc["gate_"+mode][f,a,h,v,state,env,obj,0]
            raw[mode]=np.ascontiguousarray(arc["readings_mm"][f,h,state,z,env,obj,0,None])
        expected[METHODS[0]]=np.zeros_like(expected["fixed"])
        raw["none"]=np.zeros_like(raw["fixed"])
    wall=np.zeros((5,len(METHODS)))
    order=np.zeros_like(wall,int)
    maxerr=0.
    before=snapshot()
    for rep in range(-1,len(wall)):
        seq=rng.permutation(len(METHODS))
        for m in seq:
            mode=METHODS[m]
            torch.cuda.synchronize()
            begin=time.perf_counter_ns()
            act=infer.actions43(xx,engine)
            paid=infer.paid_sensors(one,act)
            if m:
                q=np.stack([row[r.Q] for row in paid]).astype(np.float32)
                route=gates(q,raw[mode],height,mode,policy)[...,0]
            else:
                route=np.zeros_like(act,bool)
            output=timing45.decode_mixed(engine,templates[h],xx,act,paid,route)
            torch.cuda.synchronize()
            end=time.perf_counter_ns()
            if rep>=0:
                wall[rep,m]=(end-begin)*1e-6
            np.testing.assert_array_equal(act,chosen)
            np.testing.assert_array_equal(route,expected[mode])
            ref=np.where(route,pp,op)
            np.testing.assert_allclose(output,ref,atol=2e-9,rtol=0)
            maxerr=max(maxerr,float(abs(output-ref).max()))
        if rep>=0:
            order[rep]=seq
    dest=ROOT/"timing"/f"panel_{panel:02d}.npz"
    dest.parent.mkdir(exist_ok=True)
    v38.atomic_npz(dest,warm_ms=wall,order=order,methods=METHODS,rerun_max_error=maxerr,
                   route_freq=[expected[m].mean() for m in METHODS],cell=[f,a,h,v],state_counts=np.bincount(state.ravel()))
    v38.atomic_json(dest.with_suffix(".json"),dict(before=before,after=snapshot()))
    print("timed",panel,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("even","odd","timing"))
    mode=ap.parse_args().mode
    assert os.getenv("SLURM_JOB_ID") and torch.cuda.device_count()==1
    assert tuple(FAMILIES)==tuple(world48.FAMILIES)
    old.BASE=480000000
    old.SUPPORT_BASE=800
    world48.install_world43(old.world)
    torch.set_num_threads(1)
    policy=route48.load_policy("cuda")
    fixed_i=HEIGHTS.index(json.loads((ROOT/"policy48.json").read_text())["fixed_height_mm"])
    (ROOT/"runs").mkdir(exist_ok=True)
    info=dict(job=os.environ["SLURM_JOB_ID"],node=socket.gethostname(),utc_start=time.time(),
              gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
              freeze_sha256=digest(ROOT/"freeze48.sha256"))
    target=ROOT/"runs"/(mode+".json")
    v38.atomic_json(target,info)
    with torch.inference_mode():
        engine=infer.make_bank()
        templates=[[infer.make_template(engine,h*.001,u) for u in (None,AXES[str(h)]["axis"])] for h in HEIGHTS]
        for panel in (range(12) if mode=="timing" else range(int(mode=="odd"),12,2)):
            if mode=="timing":
                time_panel(panel,engine,templates,policy,fixed_i)
            else:
                one_panel(panel,engine,templates,policy,fixed_i)
    info["utc_done"]=time.time()
    v38.atomic_json(target,info)


if __name__=="__main__":
    main()
