"""Source-selected sign calibration; inputs/forecasts are [vx, vz, R*omega] in m/s."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'racket_external_v72/method'))
import bench
HERE=Path(__file__).resolve().parent
OLD=ROOT/'racket_video_v73/gpu/control'
CANDIDATES=[(h,a) for h in ('global',1.,.25) for a in (0.,.5,2.)]
MARGIN=.20
BUDGETS=(4,8)
def slip(x):
    return x[...,0]-x[...,2]
def direction(v):
    return torch.where(v>0,1.,-1.)
def inputs(x):
    sign=direction(slip(x)); normal=x[:,1].abs().clamp_min(.01)
    return torch.stack((slip(x).abs()/normal,x[:,0]*sign/normal),1)
def standardize(source):
    v=inputs(torch.cat([x for x,y in source]))
    return v.mean(0),v.std(0).clamp_min(1e-6)
def source_prior(source, tx, mean, sd):
    """One vote per source product. Class probabilities from resolved impacts only."""
    q=(inputs(tx)-mean)/sd
    votes=[]
    for x,y in source:
        ok=slip(y).abs()>MARGIN
        labels=direction(slip(y[ok]))*direction(slip(x[ok]))
        distance=torch.cdist(q,(inputs(x[ok])-mean)/sd).square()
        weight=torch.softmax(-.5*distance,1)
        votes.append(weight@labels)
    return torch.stack(votes).mean(0)
def head_inputs(source, cx, cy, idx, tx):
    mean,sd=standardize(source)
    prior=source_prior(source,tx,mean,sd)
    geom=(inputs(cx)-mean)/sd
    dt=torch.cdist((inputs(tx)-mean)/sd,geom).square().T[idx]
    sign=direction(slip(cy))*direction(slip(cx))
    return (prior,dt,sign[idx],(slip(cy[idx]).abs()>MARGIN),direction(slip(tx)))
def decide(parts,k,h,a):
    prior,dt,ys,valid,orient=parts
    w=valid[:,:k,None].double()
    if h!='global':
        w=w*torch.exp(-.5*dt[:,:k]/(h*h))
    score=(w*ys[:,:k,None]).sum(1)+a*prior
    score=torch.where(score.abs()>1e-12,score,prior)
    return direction(score)*orient
def stratified(x,cut,rng):
    slow=(x[:,1].abs()<cut).cpu().numpy()
    cal=[];hold=[]
    for mask in (slow,~slow):
        g=rng.permutation(np.flatnonzero(mask));mid=len(g)//2
        cal.extend(g[:mid]);hold.extend(g[mid:])
    return np.asarray(sorted(cal)),np.asarray(sorted(hold))
def draw_tape(x,cut,count,seed):
    fast=(x[:,1].abs()>=cut).cpu().numpy()
    if min(sum(fast),sum(~fast))<4:
        raise ValueError('insufficient speed-stratum cal samples')
    rng=np.random.default_rng(seed)
    return np.stack([np.column_stack([rng.choice(np.flatnonzero(fast),4,replace=False),rng.choice(np.flatnonzero(~fast),4,replace=False)]).ravel() for _ in range(count)])
def select(source):
    scores=[]
    for i,(name,(x,y)) in enumerate(source.items()):
        other=[p for n,p in source.items() if n!=name]
        cut=float(torch.quantile(torch.cat([p[0][:,1].abs() for p in other]),.75))
        cal,hold=stratified(x,cut,np.random.default_rng(740074+i))
        ix=torch.as_tensor(draw_tape(x[cal],cut,64,740084+i),device=x.device)
        valid=(x[hold,1].abs()>=cut)&(slip(y[hold]).abs()>MARGIN)
        tx=x[hold][valid]; target=direction(slip(y[hold][valid]))
        assert len(target)>0
        parts=head_inputs(other,x[cal],y[cal],ix,tx)
        for h,a in CANDIDATES:
            for k in BUDGETS:
                accuracy=float((decide(parts,k,h,a)==target).double().mean())
                scores.append(dict(product=name,bandwidth=h,pseudocount=a,K=k,n=len(tx),accuracy=accuracy))
    avg={c:np.mean([r['accuracy'] for r in scores if (r['bandwidth'],r['pseudocount'])==c]) for c in CANDIDATES}
    return max(CANDIDATES,key=avg.get),scores
def project(p,sign):
    before=slip(p)
    different=direction(before)!=sign
    delta=torch.where(different,sign*MARGIN-before,0.)
    new=p.clone();new[:,:,0]+=.4*delta;new[:,:,2]-=.6*delta
    assert torch.all(direction(slip(new))==sign)
    assert torch.max(((new[:,:,0]-p[:,:,0])+2/3*(new[:,:,2]-p[:,:,2])).abs())<1e-12
    return new
def numpy(t):return t.detach().cpu().numpy()
def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def evaluate(out,sources,cx,cy,tx,ty,tape,cut,choice,ids,cids,prior):
    idx=torch.as_tensor(tape,device=tx.device)
    source=list(sources.values())
    base=dict(bench.elastic_predictions(source,cx,cy,idx,tx))
    sx,sy=(torch.cat([p[j] for p in source]) for j in (0,1))
    _,expert_fit=bench.compare.old.empirical(sx,sy,sx[:1])
    cr,_,_=bench.route_predictions(cx,expert_fit);_,orig,alt=bench.route_predictions(tx,expert_fit)
    affine={k:p for k,_,p in bench.compare.affine_models(prior,cx,cy,tape,tx,[.1],True)}
    fast=tx[:,1].abs()>=cut
    parts=head_inputs(source,cx,cy,idx,tx)
    result={}; metrics=[]
    for k in BUDGETS:
        paid=idx[:,:k]
        alive=slip(cy[paid]).abs()>MARGIN
        votes=direction(slip(cy[paid]))*alive
        mismatch=(cr[paid]!=(slip(cy[paid])>0))&alive
        flip=2*mismatch.sum(1)>alive.sum(1)
        majority=torch.where(flip[:,None,None],alt,orig)
        elastic=base[k]; es=direction(slip(elastic))
        stratum=cx[paid,1].abs()>=cut
        g=votes.sum(1,keepdim=True)
        gs=torch.where(fast,(votes*stratum).sum(1,keepdim=True),(votes*(~stratum)).sum(1,keepdim=True))
        heads={'single_affine_eb':direction(slip(affine[k])),'elastic':es,'majority':direction(slip(majority)),
               'calibrated':decide(parts,k,*choice),'source_only':(direction(parts[0])*parts[-1]).expand_as(es),
               'paid_global':torch.where(g==0,es,direction(g)),
               'paid_speed':torch.where(gs==0,es,direction(gs)),
               'parity_global':decide(parts,k,'global',0.)}
        for m,sign in heads.items():
            p=affine[k] if m=='single_affine_eb' else elastic if m=='elastic' else majority if m=='majority' else project(elastic,sign)
            result[f'{m}_k{k}']=numpy(p)
            for group,inc in [('fast',fast),('all',torch.ones_like(fast)),('other',~fast)]:
                for margin in (0.,MARGIN,.5):
                    keep=(slip(ty).abs()>margin)&inc
                    if not keep.any():continue
                    v,w=bench.errors(p,ty)
                    correct=sign[:,keep]==direction(slip(ty[keep]))
                    too_far=((p[:,:,0]-tx[:,0])/(-slip(tx))).detach()
                    denom=inc&(slip(tx).abs()>1e-5)
                    metrics.append(dict(K=k,method=m,group=group,margin=margin,n=int(inc.sum()),resolved=int(keep.sum()),accuracy=float(correct.double().mean()),velocity=float(v[:,inc].mean()),spin=float(w[:,inc].mean()),disagrees_with_elastic_sign=float((sign[:,inc]!=es[:,inc]).double().mean()),alpha_outside=float(((too_far[:,denom]< -1e-9)|(too_far[:,denom]>.800000001)).double().mean())))
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'predictions.npz',target=numpy(ty),incident=numpy(tx),tape=tape,target_ids=ids,calibration_ids=cids,**result)
    write_csv(out/'metrics.csv',metrics)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('source','target'));args=ap.parse_args()
    gpu=bench.idle_device();torch.set_default_dtype(torch.float64)
    # Read only source training files during the selection phase. Avoid even loading target labels.
    src={}
    train=bench.V70/'data/upstream/train'
    def read_group(configs):
        rows=[]
        for i in configs:
            with (train/f'racket_{i:02}.csv').open() as f:rows.extend(csv.DictReader(f))
        def read(prefix,spin):
            return bench.as_cuda([[float(r[prefix+'x']),float(r[prefix+'z']),.02*float(r[spin+'y'])] for r in rows])
        return read('v_in_','spin_before_'),read('v_out_','spin_after_')
    for name,configs in bench.compare.GROUPS.items():
        if name!='abs_ii_soft':src[name]=read_group(configs)
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    meta=dict(gpu=gpu,node=platform.node(),job=os.getenv('SLURM_JOB_ID'),step=os.getenv('SLURM_STEP_ID'),torch=torch.__version__,script_sha256=digest(Path(__file__)),design_sha256=digest(HERE/'DESIGN_KO.md'))
    if args.phase=='source':
        choice,scores=select(src);write_csv(HERE/'source_validation.csv',scores)
        (HERE/'frozen.json').write_text(json.dumps(dict(selection=choice,**meta),indent=2)+'\n')
        print('selected',choice,flush=True);return
    frozen=json.loads((HERE/'frozen.json').read_text());choice=tuple(frozen['selection'])
    assert frozen['script_sha256']==meta['script_sha256'] and frozen['design_sha256']==meta['design_sha256']
    plan=np.load(OLD/'splits.npz'); old=np.load(OLD/'predictions.npz')
    x,y=read_group([10]);tx=bench.as_cuda(old['incident'][:158]);ty=bench.as_cuda(old['target'][:158])
    cut=float(torch.quantile(torch.cat([v[0][:,1].abs() for v in src.values()]),.75))
    bench.compare.BUDGETS=BUDGETS
    source_data={('train',i):read_group([i]) for n in src for i in bench.compare.GROUPS[n]}
    prior=bench.compare.make_prior(source_data,list(src),[0.,0.,0.])
    root=HERE/'result'
    res=evaluate(root/'official',src,x,y,tx,ty,plan['balanced'],cut,choice,plan['target_ids'][:158],plan['calibration_ids'],prior)
    differ={}
    for k in BUDGETS:
        for here,there in [('elastic','elastic_impulse'),('majority','majority'),('single_affine_eb','single_affine_eb')]:
            val=float(abs(res[f'{here}_k{k}']-old[f'{there}_k{k}'][:,:158]).max())
            differ[f'{here}_k{k}']=val
            assert val<1e-8
    ca,te=stratified(x,cut,np.random.default_rng(740074))
    assert not set(ca)&set(te)
    t=draw_tape(x[ca],cut,256,740075)
    evaluate(root/'secondary',src,x[ca],y[ca],x[te],y[te],t,cut,choice,plan['calibration_ids'][te],plan['calibration_ids'][ca],prior)
    (root/'run.json').write_text(json.dumps(dict(frozen_sha256=digest(HERE/'frozen.json'),exact_v73=differ,fast_cut_mps=cut,**meta),indent=2)+'\n')
if __name__=='__main__':main()
