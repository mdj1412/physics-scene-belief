"""Cold-session end-to-end GPU replay, with paired pre/post raw readings and balanced order."""
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
V50 = ROOT.parent / 'polarity_calibration_v50'
V51 = ROOT.parent / 'reference_drift_v51'
sys.path.insert(0, str(V51))
from bench51 import balanced_order, vote_weight
from bench50 import coefficients, engine50, observe, r, snapshot, unsigned_predict, update, world, world50
from policy52 import decide, load_model, witnesses

METHODS = ('likelihood_selector','vote_selector','post_vote_soft','always_unsigned')
BUDGETS = (0,1,2,4,8)
BLOCKS = 2


def choice(m, old_w, old_f, new_w, new_f, model):
    if m < 2:
        result = decide(old_w, old_f, new_w, new_f, ('likelihood','vote')[m], model)
        return result['action'], result['alpha']
    if m == 2:
        return np.ones(len(new_w),dtype=np.int8), vote_weight(new_w,new_f)
    return np.full(len(new_w),2,dtype=np.int8), np.zeros(len(new_w))


def routed_predictions(p0, displacement, arm, signed, unsigned, action, alpha):
    blind = action == 2
    out = np.empty_like(p0)
    if blind.any():
        out[blind] = unsigned_predict(p0[blind],displacement[blind],unsigned[arm[blind]])
    if (~blind).any():
        keep = ~blind
        coef = signed[arm[keep]]
        plus = update(p0[keep],displacement[keep],coef)
        minus = update(p0[keep],-displacement[keep],coef)
        a = alpha[keep,None]
        out[keep] = a*plus + (1-a)*minus
    return out


def fixture(k, model):
    rng = np.random.default_rng(520318+k)
    f,h = k%4,(k//4+k//2)%2
    alignment,variant = (k//4+k)%2,(k//2+k//4)%2
    family,height = world50.FAMILIES[f],world50.HEIGHTS[h]
    pop=world(k)
    xx,optics = r.apparatus(pop,family,height,r.VARIANTS[variant])
    optics = observe(optics,family,height,r.ALIGN[alignment],r.VARIANTS[variant])
    bought = [{key:row[key][:,:1] for key in (r.r.Q,r.r.P)} for row in optics]
    env,obj=np.indices((4,128))
    states = rng.permutation(np.repeat(np.arange(3),(256,128,128))).reshape(4,128)
    disp = np.stack([world50.readings(pop,family,height,s)[...,0] for s in world50.STATES])[states,env,obj]

    # All methods see the same measured pairs. Half of the sampled distinct environments
    # complement their retention after the physical switch; fixture metadata stay here.
    kold = (np.arange(4)+k)%5
    changed = (np.arange(4)+k)%2==0
    knew = np.where(changed,4-kold,kold)
    reps = rng.integers(256,size=4)
    full = [[] for _ in range(4)]
    for e in range(4):
        traces = witnesses(k, int(kold[e]),int(knew[e]),8)
        for bucket,array in zip(full,traces):
            bucket.append(array[e,reps[e]])
    old_w,old_f,new_w,new_f=map(np.stack,full)
    assert old_w.shape==old_f.shape==(4,4) and new_w.shape==new_f.shape==(4,8)

    name=f'panel_{k:02d}.npz'
    with np.load(V50/'forecast50'/name) as forecast:
        assert tuple(forecast['methods'].astype(str))==('original','local_plus','local_minus','fixed_plus','fixed_minus','local_unsigned_fit')
        p=forecast['p'][:,f,h,:,alignment,variant,...,0]
        plus,minus,blind=[p[j,states,env,obj] for j in (1,2,5)]
        selected=forecast['selected'][f,h,variant]
    expected=[]; order_actions=[]; order_alpha=[]
    # Call the deployment decision independently for each session in the reference path.
    for m in range(len(METHODS)):
        actions=[]; weights=[]
        for e in range(4):
            a,v=choice(m,old_w[e:e+1],old_f[e:e+1],new_w[e:e+1,:4],new_f[e:e+1,:4],model)
            actions.append(int(a[0]));weights.append(float(v[0]))
        actions=np.asarray(actions); weights=np.asarray(weights)
        expected.append(np.where(actions[:,None]==2,blind,weights[:,None]*plus+(1-weights[:,None])*minus))
        order_actions.append(actions);order_alpha.append(weights)
    return (xx,bought,disp,h,old_w,old_f,new_w,new_f,expected,selected,
            np.stack(order_actions),np.stack(order_alpha),kold,knew,reps)


def measure(k,engine,templates,signed,unsigned,model):
    (xx,bought,disp,h,ow,of,nw,nf,expected,selected,actions,alpha,kold,knew,reps)=fixture(k,model)
    gen=np.random.default_rng(520883+k)
    orders=np.concatenate([balanced_order(gen) for _ in range(BLOCKS)])
    ms=np.empty((len(orders),len(METHODS)))
    errors=np.zeros(len(METHODS))
    before=snapshot()
    for rep,seq in enumerate([gen.permutation(4),*orders],start=-1):
        for m in seq:
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            ac,weight=choice(m,ow,of,nw[:,:4],nf[:,:4],model)
            observation=r.infer.actions43(xx,engine)
            purchased=r.infer.paid_sensors(bought,observation)
            p0=r.infer.decode43(engine,templates[h],xx,observation,purchased)[...,0]
            arm=(observation==r.r.Q).astype(np.intp)
            pred=routed_predictions(p0,disp,arm,signed[1,h],unsigned[h],ac,weight)
            torch.cuda.synchronize()
            stop=time.perf_counter_ns()
            if rep>=0:ms[rep,m]=(stop-start)*1e-6
            np.testing.assert_array_equal(ac,actions[m])
            np.testing.assert_array_equal(observation,selected)
            np.testing.assert_allclose(pred,expected[m],atol=2e-9,rtol=0)
            errors[m]=max(errors[m],abs(pred-expected[m]).max())
    micro=np.empty((101,len(BUDGETS),3))
    for rep in range(-1,len(micro)):
        for packed in gen.permutation(len(BUDGETS)*3):
            j,m=divmod(packed,3)
            start=time.perf_counter_ns()
            choice(m,ow,of,nw[:,:BUDGETS[j]],nf[:,:BUDGETS[j]],model)
            if rep>=0:micro[rep,j,m]=(time.perf_counter_ns()-start)*1e-6
    path=ROOT/'timing52'/f'panel_{k:02d}.npz'
    r.v38.atomic_npz(path,ms=ms,method_order=orders,max_replay_error=errors,host_ms=micro,actions=actions,alpha=alpha,
                     pre_regime=kold,post_regime=knew,prefix_repeat=reps)
    r.v38.atomic_json(path.with_suffix('.json'),dict(before=before,after=snapshot()))
    print('finished',k,flush=True)
    return ms,errors,micro,actions


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    (ROOT/'timing52').mkdir(exist_ok=True)
    start=time.time()
    with torch.inference_mode():
        engine,templates=engine50()
        signed,unsigned=coefficients()
        model=load_model()
        measured=[measure(i,engine,templates,signed,unsigned,model) for i in range(12)]
    clock=np.stack([x[0] for x in measured])
    b=np.random.default_rng(5207714).integers(12,size=(16000,12))
    p90=np.quantile(clock,.90,axis=(0,1));boots=np.quantile(clock[b],.9,axis=(1,2))
    diff=boots[:,0,None]-boots
    acts=np.concatenate([x[3] for x in measured],axis=1)
    result=dict(methods=METHODS,p90_ms=p90.tolist(),p90_ci95=np.quantile(boots,[.025,.975],axis=0).T.tolist(),
                selector_minus_ms=(p90[0]-p90).tolist(),selector_minus_ci95=np.quantile(diff,[.025,.975],axis=0).T.tolist(),
                selector_minus_vote_and_postvote_simultaneous95=np.quantile(diff[:,1:3],[.0125,.9875],axis=0).T.tolist(),
                estimator_host_p90_ms=np.quantile(np.stack([x[2] for x in measured]),.9,axis=(0,1)).tolist(),
                host_methods=METHODS[:3],host_post_budgets=BUDGETS,source_pre_pairs=4,target_post_pairs=4,
                route_counts_per_48_sessions=[[int((a==j).sum()) for j in range(3)] for a in acts],
                max_replay_error=np.max([x[1] for x in measured],axis=0).tolist(),
                samples_per_method=clock.shape[0]*clock.shape[1],batch=512,catalogs=12,crossover='2 Williams blocks/catalog',
                job=os.getenv('SLURM_JOB_ID'),node=socket.gethostname(),gpu=torch.cuda.get_device_name(),
                torch=str(torch.__version__),cuda=torch.version.cuda,utc_start=start,utc_end=time.time())
    r.v38.atomic_json(ROOT/'timing52.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
