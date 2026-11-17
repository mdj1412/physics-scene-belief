"""Collect source-only axes and train common heads, then seal matched v66 forecasts."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
P=HERE.parent
V66=P/'invariance_audit_v66'
SRC=P/'image_pressure_v45'
LOCK=V66/'locked'
sys.path[:0]=[str(V66),str(P/'blind_decoder_v43')]
import exp_readout as read
import inference43 as inf
import world43 as world
from exp_readout import Head
SEEDS=tuple(range(670101,670105))
HEIGHTS=(19,31)
REPS=('dropped','restored')
FAMS=('wendland','dispersive')
SURF=('inlay','woven')
OLD=('rational','gaussian_rate')
NEW=('strengthening','weakening')
VAR=('base','hollow','bulge')
STAGES=('full','soft075','hard','dropped','restored')
DEV='cuda'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def name(f,s,l,v,h):return f'{f}_{s}_{l}_{v}_h{h}.npz'
def panel(k):return f'panel_{k:02d}'
def principal(rows):
    rows=rows/np.linalg.norm(rows,axis=1)[:,None]
    vals,vec=np.linalg.eigh(rows.T@rows/len(rows))
    u=vec[:,-1]
    return u*np.sign(u[np.argmax(abs(u))]),float(vals[-1])
def basis(u):
    d=np.eye(7)[-1]-u
    B=np.eye(7)-2*np.outer(d,d)/(d@d)
    np.testing.assert_allclose(B[:,-1],u,atol=2e-12)
    return B
def axes():
    hist=json.loads((SRC/'source_axis45.json').read_text())
    rec={}
    for k,seed in enumerate(SEEDS):
        rng=np.random.default_rng(seed)
        rec[seed]={}
        for h in HEIGHTS:
            diffs=[];common=[]
            for i,f in enumerate(('fold','ripple')):
                ctx=(f,'hollow',h/1000,'inlay','rational',1,750+k*4+i)
                th=world.draw_object(64,seed+100*i+h,10,world.ALPHAS[(k+i)%len(world.ALPHAS)],ctx)
                loc=world.w.view_heights(h/1000)
                a=world.contour(th,loc,ctx).cpu().numpy()
                b=world.contour(th,loc,(f,'bulge',*ctx[2:])).cpu().numpy()
                sigma=world.w.sigmas()[world.Q].cpu().numpy()
                ea,eb=(rng.standard_normal(a.shape)*sigma for _ in range(2))
                diffs.append((b+eb)-(a+ea));common.append((b+ea)-(a+ea))
            u,var=principal(np.concatenate(diffs))
            control,_=principal(np.concatenate(common))
            old=np.array(hist[str(h)]['axis'])
            assert abs(control@old)>1-2e-12
            rec[seed][h]=dict(u=u.tolist(),explained=var,cosine_to_v45=abs(float(u@old)),
               paired_noise_cosine_to_v45=abs(float(control@old)),n_pairs=128)
    dump(HERE/'axes.json',rec)
    return {h:np.array([rec[s][h]['u'] for s in SEEDS]) for h in HEIGHTS}
def forward_axis(bank,template,axes,x,q,original,mask,hist_axis,hist_hard):
    white,norm,labels=template
    us=world.t(np.vstack([axes,hist_axis]))
    coord=us@white.T
    ids=np.flatnonzero(mask.ravel())
    hard=np.tile(original[None],(len(axes),1,1,1))
    soft=hard.copy()
    err=0.
    for p in range(0,len(ids),70):
        ix=ids[p:p+70]
        view=world.t(q.reshape(-1,*q.shape[-2:])[ix])/bank.sigma[world.Q]
        prior=bank.fit(world.t(x.reshape(-1,11)[ix])).clamp_min(1e-300).log()
        logit=prior[:,None,:]+view@white.T-norm
        penalty=.5*coord[:,None,None,:].square()-(view@us.T).permute(2,0,1)[...,None]*coord[:,None,None,:]
        base=torch.softmax(logit,-1)@labels
        err=max(err,float(np.max(abs(base.cpu().numpy()-original.reshape(-1,original.shape[-1])[ix]))))
        h=(torch.softmax(logit[None]+penalty,dim=-1)@labels).cpu().numpy()
        err=max(err,float(np.max(abs(h[-1]-hist_hard.reshape(-1,hist_hard.shape[-1])[ix]))))
        hard.reshape(len(axes),-1,original.shape[-1])[:,ix]=h[:-1]
        soft.reshape(len(axes),-1,original.shape[-1])[:,ix]=(torch.softmax(logit[None]+.9375*penalty[:-1],dim=-1)@labels).cpu().numpy()
    assert err<1e-8,err
    return hard,soft,err
def raw_features(x,q,h,B,hard,mask):
    common=np.broadcast_to(x[...,None,:],(*mask.shape,11))[mask]
    obs=q[mask]*1000
    z=obs@B
    pred=np.clip(hard[mask],1e-4,1-1e-4)
    return np.column_stack([common,np.full(len(common),(h-25)/6),np.log(pred/(1-pred)),z]).astype('float32')
def features(raw,rep):
    if rep=='restored':return raw
    out=raw.clone();out[:,-1]=0
    return out
def cells(k,target,engine,U,templates,hist):
    families=FAMS if target else ('fold','ripple')
    root=LOCK if target else SRC
    for f,s,l,v,h in itertools.product(families,SURF,OLD,VAR,HEIGHTS):
        fn=name(f,s,l,v,h)
        with np.load(root/'results'/panel(k)/fn) as a:
            x,q=a['xx'],a['query_paid']
            orig,arch=a['pred'][:2,0]
            sel=a['selected']==world.Q
        n=2 if not target and k<8 else 4
        q=q[:,:,:n];orig=orig[:,:,:n];arch=arch[:,:,:n]
        mask=np.isfinite(q).all(-1)
        np.testing.assert_array_equal(mask,np.broadcast_to(sel[...,None],mask.shape))
        hard,soft,err=forward_axis(engine.inner,templates[h],U[h],x,q,orig,sel,hist[h],arch)
        raw=np.stack([raw_features(x,q,h,basis(u),hpred,mask) for u,hpred in zip(U[h],hard)])
        yield fn,raw,orig,hard,soft,mask,err
def train(fit,val,ys,vys):
    models={};records={}
    for k,seed in enumerate(SEEDS):
        tr=torch.as_tensor(np.concatenate(fit[k]),device=DEV)
        va=torch.as_tensor(np.concatenate(val[k]),device=DEV)
        for rep in REPS:
            raw=features(tr,rep);mu=raw.mean(0);sd=raw.std(0,unbiased=False).clamp_min(1e-7)
            X=(raw-mu)/sd
            V=(features(va,rep)-mu)/sd
            torch.manual_seed(66101)
            net=Head().cuda(); opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
            gen=torch.Generator(device=DEV).manual_seed(66111)
            best=float('inf'); trace={}
            fn=HERE/f'{seed}_{rep}.pt'
            for e in range(1,17):
                for g in torch.randperm(len(ys),device=DEV,generator=gen).split(4096):
                    p=net(X[g],tr[g,12]);loss=(p-ys[g]).square().mean()
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if e in (4,8,16):
                    with torch.no_grad(): risk=float(read.risk(net(V,va[:,12]),vys))
                    trace[e]=risk
                    if risk<best:
                        best=risk;trace['best_epoch']=e
                        torch.save(dict(weights=net.state_dict(),mu=mu,sd=sd),fn)
            state=torch.load(fn,map_location=DEV,weights_only=True);net.load_state_dict(state['weights'])
            models[k,rep]=net.eval(),mu,sd
            records[f'{seed}_{rep}']=dict(validation=trace,sha256=digest(fn))
    dump(HERE/'fit.json',records)
    return models
def main():
    assert torch.cuda.device_count()==1 and os.environ['SLURM_JOB_ID'] and not (HERE/'labels').exists()
    torch.set_num_threads(1)
    U=axes()
    h0=json.loads((SRC/'source_axis45.json').read_text())
    hist={h:np.array(h0[str(h)]['axis']) for h in HEIGHTS}
    engine=inf.make_bank()
    templates={h:inf.make_template(engine,h/1000) for h in HEIGHTS}
    data=[[[] for _ in SEEDS] for _ in range(2)];ys=[[],[]];maxerr=0.
    with torch.no_grad():
        for k in range(12):
            val=int(k>=8)
            for fn,raw,_,_,_,mask,err in cells(k,False,engine,U,templates,hist):
                y=read.labels(SRC/'labels45'/panel(k)/fn)
                ys[val].append(np.broadcast_to(y[...,None],mask.shape)[mask])
                for i in range(len(SEEDS)):data[val][i].append(raw[i])
                maxerr=max(maxerr,err)
            print('source',k,flush=True)
    y=[torch.tensor(np.concatenate(s),device=DEV,dtype=torch.float32) for s in ys]
    models=train(*data,*y)
    del data,y,ys
    out=HERE/'forecasts';out.mkdir(exist_ok=True)
    shards={}
    with torch.no_grad():
        for k in range(12):
            rows=[];names=[]
            for fn,raw,orig,hard,soft,mask,err in cells(k,True,engine,U,templates,hist):
                p=np.zeros((len(SEEDS),len(STAGES),*orig.shape),np.float32)
                for i in range(len(SEEDS)):
                    p[i]=np.broadcast_to(orig,p[i].shape)
                    p[i,1],p[i,2]=soft[i],hard[i]
                    a=torch.as_tensor(raw[i],device=DEV)
                    for j,rep in enumerate(REPS,3):
                        net,mu,sd=models[i,rep]
                        p[i,j][mask]=net((features(a,rep)-mu)/sd,a[:,12]).cpu().numpy()
                assert np.isfinite(p).all() and np.min(p)>=0 and np.max(p)<=1
                rows.append(p);names.append(fn);maxerr=max(maxerr,err)
            path=out/(panel(k)+'.npz')
            np.savez_compressed(path,p=np.stack(rows),cells=names)
            shards[path.name]=digest(path)
            print('forecast',k,flush=True)
    dump(HERE/'forecast_seal.json',dict(shards=shards,axes=digest(HERE/'axes.json'),fits=digest(HERE/'fit.json'),
           plan=digest(HERE/'plan.sha256'),max_archive_endpoint_error=maxerr,utc=time.time(),device=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
