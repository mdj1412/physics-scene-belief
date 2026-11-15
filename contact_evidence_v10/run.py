import argparse,hashlib,json,os,platform,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model import *

ROOT=Path('results')
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()
def timed(fn):
    sync();start=time.perf_counter();out=fn();sync();return out,time.perf_counter()-start
def paths(split):return sorted((ROOT/split/'public').glob('*.json'))

def prepare(split,n):
    seed={'pilot':103701,'development':203701,'test':303701}[split]
    for i in range(n):
        rng=np.random.default_rng(seed+i*137);g,e=prior_draw(rng,1);g=g[0];e=float(e[0]);side=int(rng.integers(2))
        visible=(FY<=.006) if side==0 else (FY>=-.006)
        ys=FY[visible];raw,feat,clean=sensor(g,ys,F_DX,F_W,F_PIXEL_SD,rng)
        pub=dict(id=i,ys=ys.tolist(),features=feat.tolist(),e_observed=float(e+rng.normal(0,.06)),visible_side=side)
        save(ROOT/split/'public'/f'{i:03d}.json',pub)
        actual={}
        for a,ay in zip(['left_view','right_view'],VY):
            actual[a]=np.stack([sensor(g,ay,V_DX,V_W,V_PIXEL_SD,rng)[1] for _ in range(16)])
        actual['probe']=e+rng.normal(0,.015,16)
        p=ROOT/split/'private'/f'{i:03d}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        twin=g.copy();twin[2 if side==0 else 1]*=-1
        twin_clean=sensor(twin,ys,F_DX,F_W,0,np.random.default_rng(0))[2]
        np.savez_compressed(p,g=g,e=e,twin=twin,target=response(g,e),twin_target=response(twin,e),
                            factual_equal_max=np.max(abs(clean-twin_clean)),**actual)
        if i<4:
            a=ROOT/split/'examples';a.mkdir(exist_ok=True)
            np.savez_compressed(a/f'{i:03d}_raw.npz',raw=raw,clean=clean,ys=ys,features=feat)
            vis=np.ones((81,F_W,3))*.5;vis[visible]=np.clip(raw[:,:,None],0,1)
            Image.fromarray((vis*255).astype('uint8')).resize((384,648)).save(a/f'{i:03d}_factual.png')
    save(ROOT/split/'generation.json',dict(n=n,seed=seed,repeats=16,queries=QUERIES.tolist(),time=time.time()))
    print('prepared',split,n,flush=True)

def acquire(split):
    quad=Quad(64);rules=load(ROOT/'rules.json') if (ROOT/'rules.json').exists() else None
    for p in paths(split):
        dest=ROOT/split/'policies'/p.name
        if dest.exists():continue
        public=load(p);i=public['id'];J,h,ep,eh=info(public)
        current=quad.predict(J,h,ep,eh)[0]
        (g,e),bt=timed(lambda:draw(J,h,ep,eh,32768,500001+i*97))
        og,oe=draw(J,h,ep,eh,8192,600001+i*97)
        outer=hypothetical(og,oe,256,700001+i*97)
        ids=torch.cat([torch.arange(4096,device=DEV),torch.arange(16384,20480,device=DEV)])
        (s8,c8,ess8,eig8),tm8=timed(lambda:utilities(g[ids],e[ids],outer,current))
        (s32,c32,ess32,eig32),tm32=timed(lambda:utilities(g,e,outer,current))
        def exact():
            vals=[np.zeros(6)]
            for a in ACTIONS[1:]:
                jj,hh,pp,ee=update_batch(J,h,ep,eh,a,outer[a])
                post=quad.predict(jj,hh,pp,ee)
                vals.append((.5*((post-current)**2).sum(2).mean(0)).cpu().numpy())
            return np.stack(vals,1)
        uq,qt=timed(exact)
        jc,hc,pc,ec=info(public,'coarse');cm=quad.predict(jc,hc,pc,ec)[0]
        cg,ce=draw(jc,hc,pc,ec,8192,800001+i*97);cog,coe=draw(jc,hc,pc,ec,8192,900001+i*97)
        co=hypothetical(cog,coe,256,1000001+i*97,'coarse')
        (_,uc,_,_),ct=timed(lambda:utilities(cg,ce,co,cm,'coarse'))
        # Sensitivity to outer MC, with a separate H512 generator; audit first16.
        audit={}
        if i<16:
            oo=hypothetical(og,oe,512,1100001+i*97)
            ss,cc,_,_=utilities(g,e,oo,current)
            audit['cf32_h512']=cc.tolist()
            qo=Quad(96);u96=[np.zeros(6)]
            for a in ACTIONS[1:]:
                jj,hh,pp,ee=update_batch(J,h,ep,eh,a,outer[a]);post=qo.predict(jj,hh,pp,ee)
                u96.append((.5*((post-current)**2).sum(2).mean(0)).cpu().numpy())
            audit['quad96_same_outer']=np.stack(u96,1).tolist()
        utilities_dict={'boundary_standard8192':s8,'boundary_cf8192':c8,'boundary_cf32768':c32,'boundary_quad64':uq,'coarse_cf8192':uc}
        choices={k:v.argmax(1).tolist() for k,v in utilities_dict.items()}
        choices.update(none=[0]*6,left_view=[1]*6,right_view=[2]*6,probe=[3]*6,plugin_certainty=[3]*6)
        # EIG is empirical KL posterior/prior on the finite bank, descriptive only.
        choices['boundary_eig']=[int(np.argmax(eig32))+1]*6
        if split=='test':
            if rules is None:raise RuntimeError('Development rules must be frozen before test.')
            choices['query_type_rule']=[rules['query_type']]*6
            choices['query_value_rule']=rules['query_value']
            choices['visibility_query_rule']=[rules['visibility_query'][str(public['visible_side'])][q] for q in range(6)]
        save(dest,dict(id=i,public_sha=digest(p),choices=choices,utilities={k:v.tolist() for k,v in utilities_dict.items()},
                       seconds=dict(bank32768=bt,standard_and_cf8192=tm8,standard_and_cf32768=tm32,quad64=qt,coarse8192=ct),
                       ess8192=ess8,ess32768=ess32,audit=audit))
        print('acquired',split,i,flush=True)
    files=sorted((ROOT/split/'policies').glob('*.json'))
    save(ROOT/split/'freeze.json',dict(count=len(files),time=time.time(),policies={p.name:digest(p) for p in files},
                                     rules_sha=digest(ROOT/'rules.json') if rules else None))

def evaluate(split):
    freeze=load(ROOT/split/'freeze.json');quad=Quad(64);quad96=Quad(96)
    if freeze['count']!=len(paths(split)):raise RuntimeError('Incomplete policy freeze')
    for p in paths(split):
        pp=ROOT/split/'policies'/p.name
        if digest(pp)!=freeze['policies'][p.name]:raise RuntimeError('Policy changed')
        policy=load(pp)
        if digest(p)!=policy['public_sha']:raise RuntimeError('Public data changed')
        dest=ROOT/split/'evaluation'/p.with_suffix('.npz').name
        if dest.exists():continue
        public=load(p);i=public['id'];truth=np.load(ROOT/split/'private'/dest.name)
        preds=[];coarse=[];plugin=[];audit=0.
        for mode,out in [('boundary',preds),('coarse',coarse)]:
            J,h,ep,eh=info(public,mode)
            base=quad.predict(J,h,ep,eh)[0].cpu().numpy();out.append(np.repeat(base[None],16,axis=0))
            if mode=='boundary':
                _,am=quad.predict(J,h,ep,eh,True);pg=torch.cat([torch.zeros((1,1),device=DEV,dtype=DT),am],1)
                pv=predictions(pg,trunc_mean(ep,eh).reshape(1))[0].cpu().numpy();plugin.append(np.repeat(pv[None],16,axis=0))
            for a in ACTIONS[1:]:
                jj,hh,ppp,ee=update_batch(J,h,ep,eh,a,truth[a],mode)
                pv,am=quad.predict(jj,hh,ppp,ee,True);out.append(pv.cpu().numpy())
                if mode=='boundary':
                    pg=torch.cat([torch.zeros((len(am),1),device=DEV,dtype=DT),am],1)
                    em=trunc_mean(ppp,ee).expand(len(am));plugin.append(predictions(pg,em).cpu().numpy())
                    if i<16:
                        reference=quad96.predict(jj,hh,ppp,ee)
                        audit=max(audit,float((pv-reference).abs().max()))
        pred=np.stack(preds);cp=np.stack(coarse);pl=np.stack(plugin);target=truth['target']
        losses=.5*((pred-target)**2).sum(-1)
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,prediction=pred,losses=losses,coarse_losses=.5*((cp-target)**2).sum(-1),
                            plugin_losses=.5*((pl-target)**2).sum(-1),target=target,
                            twin_target=truth['twin_target'],factual_equal_max=truth['factual_equal_max'],
                            quadrature_prediction_maxdiff=audit)
        print('evaluated',split,i,flush=True)
    if split=='development':
        arrays=[np.load(ROOT/split/'evaluation'/p.with_suffix('.npz').name)['losses'].mean(1) for p in paths(split)]
        loss=np.stack(arrays);mean=loss.mean(0)
        sides=np.array([load(p)['visible_side'] for p in paths(split)])
        save(ROOT/'rules.json',dict(query_type=int(mean.mean(1).argmin()),query_value=mean.argmin(0).tolist(),
                                   visibility_query={str(s):loss[sides==s].mean(0).argmin(0).tolist() for s in [0,1]},
                                   development_worlds=len(loss),time=time.time()))

def validate():
    rng=np.random.default_rng(193);g,e=prior_draw(rng,100)
    maxarea=0.;maxnormal=0.;maxenergy=0.
    for gg,ee in zip(g,e):
        raw,feat,clean=sensor(gg,FY,F_DX,F_W,0,rng)
        maxarea=max(maxarea,float(abs(feat-design(FY)@gg).max()))
        d=1e-7;s=(design(QUERIES+d)@gg-design(QUERIES-d)@gg)/(2*d)
        maxnormal=max(maxnormal,float(abs(s-QB@gg[1:]).max()))
        v=response(gg,ee);maxenergy=max(maxenergy,float(np.max((v*v).sum(-1)-1)))
    p=dict(ys=FY[FY<.006].tolist(),features=(design(FY[FY<.006])@g[0]).tolist(),e_observed=float(e[0]))
    J,h,ep,eh=info(p);q=Quad(64);m=q.predict(J,h,ep,eh)
    q96=Quad(96).predict(J,h,ep,eh)
    gg,ee=draw(J,h,ep,eh,131072,982)
    mc=predictions(gg,ee).mean(0)
    result=dict(image_area_error=maxarea,finite_difference_normal_error=maxnormal,energy_excess=maxenergy,
                quad64_96=float(abs(m-q96).max()),monte_carlo_reference_maxdiff=float(abs(m[0]-mc).max()),
                collision_vx_max=float(predictions(t(g),t(e))[:,:,0].max()))
    assert maxarea<1e-12 and maxnormal<1e-8 and maxenergy<1e-12
    assert result['quad64_96']<1e-7 and result['monte_carlo_reference_maxdiff']<.004
    assert result['collision_vx_max']<0
    save(ROOT/'validation.json',result);print(result,flush=True)

def main():
    torch.set_num_threads(2)
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Must run on allocated compute node')
    ROOT.mkdir(exist_ok=True)
    command=sys.argv[1]
    if command=='validate':validate();return
    if command=='prepare':prepare(sys.argv[2],int(sys.argv[3]));return
    if command=='acquire':acquire(sys.argv[2]);return
    if command=='evaluate':evaluate(sys.argv[2]);return
    if command=='pipeline':
        save(ROOT/'environment.json',dict(host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),device=DEV,
             gpu=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),time=time.time()))
        save(ROOT/'source_hashes.json',{p.name:digest(p) for p in Path('.').glob('*.py')})
        validate();prepare('pilot',8);acquire('pilot');evaluate('pilot')
        prepare('development',32);acquire('development');evaluate('development')
        prepare('test',128);acquire('test');evaluate('test')
        save(ROOT/'complete.json',dict(time=time.time(),test_worlds=128,development_worlds=32,pilot_worlds=8))
        return
    raise ValueError(command)
if __name__=='__main__':main()
