"""Factorial forecasts and pixel routing without reading target labels."""
import argparse
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np
from population47 import FAMILIES, HEIGHTS, OFFSETS, ROOT, PROJECT, old, r, infer, v38, torch

sys.path.insert(0, str(PROJECT / "normal_variance_v46"))
import camera46
import timing45
import world47
import route47

AXES = json.loads((PROJECT/"image_pressure_v45"/"source_axis45.json").read_text())
METHODS = ("always_original", "v46", "observation_only")
VARIANTS = ("hollow", "bulge")
ALIGN = ("low", "high")


def read_world(panel):
    with np.load(ROOT/"worlds"/f"panel_{panel:02d}.npz") as src:
        return {k: src[k] for k in src}


def apparatus(pop, family, height, variant):
    """Observed test apparatus only. The future intervention does not occur here."""
    xs, sensors = [], []
    for e in range(4):
        ctx = (family, variant, height*.001, str(pop["surface"][e]), str(pop["law"][e]),
               int(pop["signs"][e]), int(pop["support_panel"]))
        ds = int(pop["seed"]) + 200 + e
        theta = old.world.t(pop["theta"][e])
        xs.append(old.world.factual(theta, ds, 21, ctx))
        response = old.world.impact(theta, 0., old.world.w.PROBE_INCOMING, ctx, str(pop["regimes"][e]))[0]
        means = {r.Q: old.world.contour(theta, old.world.w.view_heights(height*.001), ctx), r.P: response}
        row = {}
        for sensor, mu in means.items():
            eps = old.world.w.stream(ds, 40+sensor).normal(size=(128, 4, mu.shape[-1]))
            value = mu[:, None] + old.world.t(eps)*old.world.w.sigmas()[sensor]
            row[sensor] = value.cpu().numpy()
        sensors.append(row)
    return np.stack(xs), sensors


def add_axis(sensors, family, height, alignment, variant):
    delta = world47.sensor_delta_mm(family, height, alignment, variant) * .001
    assert np.shape(delta) == (7,)
    return [{r.Q: s[r.Q] + delta, r.P: s[r.P]} for s in sensors]


def clip_seed(panel, h, offset_i):
    # Identical optical nuisance for paired families, variants and alignments.
    return 470100413 + panel*1000 + h*10 + offset_i


def render(pop, f, h, v, offsets, seed):
    return world47.render_clip(pop["theta"].reshape(512, 6), FAMILIES[f], VARIANTS[v], HEIGHTS[h],
                               offsets, seed, device="cpu").contiguous()


def camera_features(clip):
    frames = clip.to("cuda")
    x = camera46.infer_x(frames).reshape(4, 128)
    var = camera46.infer_normal_variance(frames, x.flatten()).reshape_as(x)
    return x, var


def paid_gate(paid, x, var, height, policy):
    query = torch.as_tensor(np.stack([s[r.Q] for s in paid]), device="cuda")
    decision = route47.decision(query, x[..., None], var[..., None], height, policy)
    return decision.cpu().numpy()


def one_panel(panel, engine, templates, lookup, policy):
    pop = read_world(panel)
    # Forecasts do not depend on contact offset or mechanical coupling. Route arrays
    # keep offset and noise axes exactly where observable data change.
    p_orig = np.empty((4,2,2,2,4,128,4),np.float64)
    p_proj = np.empty_like(p_orig)
    gate_new = np.empty((4,2,2,2,3,4,128,4),bool)
    gate_old = np.empty((4,2,2,3,4,128),bool)
    x_est = np.empty(gate_old.shape,np.float32)
    v_est = np.empty_like(x_est)
    selected = np.empty((4,2,2,4,128),np.int8)
    queries = np.empty(p_orig.shape+(7,),np.float32)
    cache = {}
    for f,family in enumerate(FAMILIES):
        for h,height in enumerate(HEIGHTS):
            for v,variant in enumerate(VARIANTS):
                xx, obs = apparatus(pop, family, height, variant)
                key = hashlib.sha256(xx.tobytes()).digest()
                if key not in cache:
                    cache[key] = infer.actions43(xx,engine)
                act = cache[key]
                selected[f,h,v] = act
                geometry = []
                for z,offset in enumerate(OFFSETS):
                    clip = render(pop,f,h,v,np.full(512,offset),clip_seed(panel,h,z))
                    x,var = camera_features(clip)
                    geometry.append((x,var))
                    x_est[f,h,v,z] = x.cpu().numpy()
                    v_est[f,h,v,z] = var.cpu().numpy()
                    gate_old[f,h,v,z] = camera46.gate46(x,var,lookup).cpu().numpy()
                for a,alignment in enumerate(ALIGN):
                    paid = infer.paid_sensors(add_axis(obs,family,height,alignment,variant),act)
                    for k,buf in enumerate((p_orig,p_proj)):
                        buf[f,a,h,v] = infer.decode43(engine,templates[h][k],xx,act,paid)
                    np.testing.assert_allclose(p_orig[f,a,h,v][act==r.P],p_proj[f,a,h,v][act==r.P],atol=1e-12,rtol=0)
                    queries[f,a,h,v] = np.stack([s[r.Q] for s in paid])
                    for z,(x,var) in enumerate(geometry):
                        gate_new[f,a,h,v,z] = paid_gate(paid,x,var,height,policy)
        print("forecast family",panel,family,flush=True)
    assert np.isfinite(p_orig).all() and np.isfinite(p_proj).all()
    assert min(p_orig.min(),p_proj.min())>=-1e-12 and max(p_orig.max(),p_proj.max())<=1+1e-12
    dest = ROOT/"forecast"/f"panel_{panel:02d}.npz"
    dest.parent.mkdir(exist_ok=True)
    v38.atomic_npz(dest,p_original=p_orig,p_projected=p_proj,gate_v46=gate_old,gate_new=gate_new,
                   x_mm=x_est,var_rad2=v_est,selected=selected,query_paid=queries,
                   families=FAMILIES,alignments=ALIGN,variants=VARIANTS,heights_mm=HEIGHTS,offsets_mm=OFFSETS,
                   query_id=r.Q,probe_id=r.P,panel=panel)
    v38.atomic_json(dest.with_suffix(".json"),{"sha256":v38.digest(dest),"n_action_groups":len(cache)})


def time_panel(panel,engine,templates,lookup,policy):
    from timing_repeat45 import snapshot
    pop = read_world(panel)
    f = panel % 4
    h = (panel//4 + panel//2) % 2
    a = (panel//4 + panel) % 2
    v = (panel//2 + panel//4) % 2
    height = HEIGHTS[h]
    xx, base = apparatus(pop,FAMILIES[f],height,VARIANTS[v])
    obs = add_axis(base,FAMILIES[f],height,ALIGN[a],VARIANTS[v])
    one = [{k:s[k][:,:1] for k in (r.P,r.Q)} for s in obs]
    # Rebuild the exact same raw input as scoring. A mixture of the three offsets
    # in a batch prevents a constant camera route from gaining a timing advantage.
    rng = np.random.default_rng(470099201+panel)
    zarr = rng.permutation(np.resize(np.arange(3),512))
    clips = [render(pop,f,h,v,np.full(512,z),clip_seed(panel,h,k)) for k,z in enumerate(OFFSETS)]
    clip = torch.stack(clips,0)[zarr,np.arange(512)].contiguous()
    with np.load(ROOT/"forecast"/f"panel_{panel:02d}.npz") as res:
        original=res["p_original"][f,a,h,v,...,0]
        projected=res["p_projected"][f,a,h,v,...,0]
        select_ref=res["selected"][f,h,v]
        old_g=res["gate_v46"][f,h,v].reshape(3,512)[zarr,np.arange(512)].reshape(4,128)
        new_g=res["gate_new"][f,a,h,v,...,0].reshape(3,512)[zarr,np.arange(512)].reshape(4,128)
    wall=np.zeros((5,3)); order=np.zeros_like(wall,int)
    maxerr=0.
    before=snapshot()
    for rep in range(-1,len(wall)):
        seq=rng.permutation(3)
        for m in seq:
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            act=infer.actions43(xx,engine)
            paid=infer.paid_sensors(one,act)
            if m==0:
                gate=np.zeros((4,128),bool)
            else:
                x,var=camera_features(clip)
                if m==1:
                    gate=camera46.gate46(x,var,lookup).cpu().numpy()
                else:
                    gate=paid_gate(paid,x,var,height,policy)[...,0]
            output=timing45.decode_mixed(engine,templates[h],xx,act,paid,gate)
            torch.cuda.synchronize()
            end=time.perf_counter_ns()
            if rep>=0:wall[rep,m]=(end-start)*1e-6
            np.testing.assert_array_equal(act,select_ref)
            if m:np.testing.assert_array_equal(gate,(old_g,new_g)[m-1])
            reference=np.where(gate,projected,original)
            np.testing.assert_allclose(output,reference,atol=2e-9,rtol=0)
            maxerr=max(maxerr,float(abs(output-reference).max()))
        if rep>=0:order[rep]=seq
    target=ROOT/"timing"/f"panel_{panel:02d}.npz"
    target.parent.mkdir(exist_ok=True)
    v38.atomic_npz(target,warm_ms=wall,order=order,methods=METHODS,rerun_max_error=maxerr,
                   routes=np.array([0.,old_g.mean(),new_g.mean()]),cell=[f,a,h,v],input_bytes=clip.numel())
    v38.atomic_json(target.with_suffix(".json"),dict(before=before,after=snapshot()))
    print("timing",panel,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("even","odd","timing"))
    mode=parser.parse_args().mode
    assert os.environ.get("SLURM_JOB_ID") and torch.cuda.device_count()==1
    assert FAMILIES==world47.FAMILIES and ALIGN==world47.ALIGNMENTS and VARIANTS==world47.VARIANTS
    world47.install_world43(old.world)
    torch.set_num_threads(1)
    lookup=camera46.load_lookup("cuda")
    policy=route47.load_policy("cuda")
    run_dir=ROOT/"runs";run_dir.mkdir(exist_ok=True)
    info=dict(job=os.environ["SLURM_JOB_ID"],node=socket.gethostname(),utc_start=time.time(),
              gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
              freeze_sha256=v38.digest(ROOT/"freeze47.sha256"))
    v38.atomic_json(run_dir/(mode+".json"),info)
    with torch.inference_mode():
        engine=infer.make_bank()
        templates=[[infer.make_template(engine,h*.001,u) for u in (None,AXES[str(h)]["axis"])] for h in HEIGHTS]
        for panel in (range(12) if mode=="timing" else range(int(mode=="odd"),12,2)):
            if mode=="timing":
                time_panel(panel,engine,templates,lookup,policy)
            else:
                one_panel(panel,engine,templates,lookup,policy)
    info["utc_done"]=time.time();v38.atomic_json(run_dir/(mode+".json"),info)


if __name__=="__main__":
    main()
