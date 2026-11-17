"""Blind GPU prediction from acquired tapes. No terminal solver or scoring import."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

H=Path(__file__).resolve().parent
P=H.parent
V67=P/'law_axis_v67'
LOCK=P/'invariance_audit_v66'/'locked'
sys.path.insert(0,str(V67))
import forecast67 as frozen

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def load_source():
    params=json.loads((V67/'axes.json').read_text())
    axes={h:np.array([params[str(seed)][str(h)]['u'] for seed in frozen.SEEDS]) for h in frozen.HEIGHTS}
    heads={}
    weights=json.loads((V67/'fit.json').read_text())
    for i,seed in enumerate(frozen.SEEDS):
        for rep in frozen.REPS:
            key=f'{seed}_{rep}';p=V67/(key+'.pt')
            assert digest(p)==weights[key]['sha256']
            z=torch.load(p,map_location='cuda',weights_only=True)
            net=frozen.Head().cuda()
            net.load_state_dict(z['weights']);heads[i,rep]=net.eval(),z['mu'],z['sd']
    return axes,heads

def physics(engine,template,axis,x,q,p,act):
    # Public height determines template. The boolean action mask restricts the only observation read.
    tr=frozen.world.t
    bank=engine.inner
    c,n,labels=template
    us=tr(axis);coord=us@c.T
    buy=act.ravel()==frozen.world.Q; nrep=q.shape[-2]
    W=len(axis)
    pred=np.empty((W,5,act.size,nrep),np.float32)
    for flag,data in [(True,q),(False,p)]:
        paid=data.reshape(act.size,nrep,-1)
        ind=np.flatnonzero(buy==flag)
        assert np.isfinite(paid[ind]).all() and np.isnan(paid[buy!=flag]).all()
        for j in range(0,len(ind),64):
            ix=ind[j:j+64]
            prior=bank.fit(tr(x.reshape(-1,11)[ix]))[:,None]
            obs=tr(paid[ix])
            if flag:
                view=obs/bank.sigma[frozen.world.Q]
                logit=prior.clamp_min(1e-300).log()+view@c.T-n
                base=torch.softmax(logit,-1)@labels
                penalty=.5*coord[:,None,None,:].square()-(view@us.T).permute(2,0,1)[...,None]*coord[:,None,None,:]
                soft=torch.softmax(logit[None]+.9375*penalty,-1)@labels
                hard=torch.softmax(logit[None]+penalty,-1)@labels
                for stage,values in enumerate((base[None].expand(W,-1,-1),soft,hard,hard,hard)):
                    pred[:,stage][:,ix]=values.cpu().numpy()
            else:
                result=(bank.update(prior,frozen.world.P,obs)@labels).cpu().numpy()
                pred[:,:,ix,:]=result[None,None]
    return pred.reshape(W,5,*act.shape,nrep)

def predict(x,q,p,act,height,engine,templ,axis,heads):
    mask=np.isfinite(q).all(-1)
    pred=physics(engine,templ,axis,x,q,p,act)
    for i,u in enumerate(axis):
        raw=torch.as_tensor(frozen.raw_features(x,q,height,frozen.basis(u),pred[i,2],mask),device='cuda')
        for stage,rep in enumerate(frozen.REPS,3):
            net,mu,sd=heads[i,rep]
            pred[i,stage][mask]=net((frozen.features(raw,rep)-mu)/sd,raw[:,12]).cpu().numpy()
    assert np.isfinite(pred).all() and np.min(pred)>=0 and np.max(pred)<=1
    return pred

def shifts(x,y,d,artifact):
    from continuous38 import source_optimum
    beta=frozen.inf.r.v38.BETA
    shifts=[]
    for final in [None,y[:,3]]:
        z,w=source_optimum(artifact,x,y[:,:3],d[:,:3],final)
        shifts.append(beta*z*w)
    return shifts

def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    assert not (H/'FORECASTS_SEALED').exists()
    axes,heads=load_source()
    engine=frozen.inf.make_bank();temps={h:frozen.inf.make_template(engine,h/1000.) for h in frozen.HEIGHTS}
    artifact=frozen.inf.r.v38.load_artifacts()[1.1]
    torch.set_num_threads(1)
    began=time.time()
    inputs={};outputs={};audit={}
    with torch.inference_mode():
        for backend in ('prescribed','compliant'):
            (H/'forecasts'/backend).mkdir(parents=True,exist_ok=True)
            for k in range(12):
                parts=[];s3=[];s4=[];cells=[];actions=[];probe_changes=[];cache={}
                directory=H/'tapes'/backend/frozen.panel(k)
                files=sorted(directory.glob('*.npz'));assert len(files)==48,(backend,k,len(files))
                for source in files:
                    inputs[str(source.relative_to(H))]=digest(source)
                    with np.load(source) as z:
                        x=z['xx'];q=z['query'];p=z['probe'];cue3,cue4=shifts(z['cx'],z['cy'],z['dirs'],artifact)
                    h=int(source.stem.rsplit('_h',1)[-1])
                    cachekey=hashlib.sha256(x.tobytes()).digest()
                    if cachekey not in cache:cache[cachekey]=frozen.inf.actions43(x,engine)
                    act=cache[cachekey]
                    # Blueprints are known to simulator, policy receives exactly one sensor.
                    q[act!=frozen.world.Q]=np.nan;p[act!=frozen.world.P]=np.nan
                    probs=predict(x,q,p,act,h,engine,temps[h],axes[h],heads)
                    if backend=='prescribed' and k==0 and not parts:
                        # Restore the original evidence and source checkpoints; verify every stage
                        # against the stored v67 forecast from before the v68 intervention.
                        name=source.name.replace('strengthening','rational').replace('weakening','rational')
                        with np.load(LOCK/'results'/frozen.panel(k)/name) as z0:
                            zx=z0['xx'];qa=z0['query_paid'];pa=z0['probe_paid'];sa=z0['selected']
                        histor=predict(zx,qa,pa,sa,h,engine,temps[h],axes[h],heads)
                        with np.load(V67/'forecasts'/(frozen.panel(k)+'.npz')) as z0:
                            ref=z0['p'][list(z0['cells']).index(name)]
                        err=float(np.abs(histor-ref).max())
                        assert err<5e-5,err
                        audit['historical_replay_max_probability_error']=err
                    parts.append(probs);cells.append(source.name);s3.append(cue3);s4.append(cue4);actions.append(act)
                path=H/'forecasts'/backend/(frozen.panel(k)+'.npz')
                np.savez_compressed(path, p=np.stack(parts),shift=s4,shift3=s3,cells=cells,actions=actions)
                outputs[str(path.relative_to(H))]=digest(path)
                print('forecast',backend,k,flush=True)
    audit['duration_s']=time.time()-began;audit['device']=torch.cuda.get_device_name()
    audit['job']=os.environ['SLURM_JOB_ID'];audit['started_unix']=began
    (H/'gpu_run.json').write_text(json.dumps(audit,indent=2))
    (H/'tapes.sha256').write_text(''.join(f'{h}  {p}\n' for p,h in inputs.items()))
    seals=dict(outputs=outputs, inputs=digest(H/'tapes.sha256'),source_heads=digest(V67/'fit.json'),
               source_axes=digest(V67/'axes.json'), source_plan=digest(H/'forecast_plan.sha256'), forecast_code=digest(Path(__file__)),
               utc=time.time(),device=audit['device'],pid=os.getpid())
    for file,sha in outputs.items():assert digest(H/file)==sha
    (H/'FORECASTS_SEALED').write_text(json.dumps(seals,indent=2)+'\n')
if __name__=='__main__':main()
