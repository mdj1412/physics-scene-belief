import os,json,time,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from model import *

R=Path('results')
SPLITS={'pilot':(8,51001),'train':(256,61001),'validation':(128,71001),'test':(256,81001),'misspec':(128,91001)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2))
def prepare():
    for split,(n,seed) in SPLITS.items():
        (R/split/'private').mkdir(parents=True,exist_ok=True)
        for i in range(n):
            p,truth=draw_truth(seed+i*137,split=='misspec')
            write(R/split/'public'/f'{i:04d}.json',p)
            np.savez_compressed(R/split/'private'/f'{i:04d}.npz',**truth)
    write(R/'source_start.json',{p.name:sha(p) for p in Path('.').glob('*') if p.is_file()})
    write(R/'environment.json',dict(time=time.time(),host=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],torch=torch.__version__,gpu=torch.cuda.get_device_name(),python=sys.version))

def pilot():
    results={};prev=None
    for n in [21,31,41,61]:
        b=Bank(n);pred=[];risks=[];started=time.time()
        for i in range(8):
            p=json.loads((R/'pilot/public'/f'{i:04d}.json').read_text());w=b.posterior(p)
            truth=np.load(R/'pilot/private'/f'{i:04d}.npz')
            mm,pp=b.decode(w);pred.extend([mm.cpu().numpy(),pp.cpu().numpy()])
            for a in [1,2,3]:
                ww=b.update(w,a,truth['obs'+str(a)][:8]);mm,pp=b.decode(ww);pred.extend([mm.cpu().numpy(),pp.cpu().numpy()])
            outer=b.outer(w,128,1000+i);jr,pr,_=b.risks(w,outer);risks.append(jr)
        vector=np.concatenate([x.ravel() for x in pred]);j=np.array(risks)
        results[str(n)]=dict(seconds=time.time()-started,max_prediction_change=None if prev is None else float(abs(vector-prev).max()),mean_prediction_change=None if prev is None else float(abs(vector-prev).mean()),actions=np.argmin(j,1).tolist(),risk=j.tolist())
        prev=vector;del b;torch.cuda.empty_cache();write(R/'pilot_audit.json',results);print(n,results[str(n)],flush=True)
    # Physics properties, independent of posterior integration.
    b=Bank(21);z=b.z;th=b.theta
    d=1e-7;numeric=(profile(th,np.array([L+d]))-profile(th,np.array([L-d]))).flatten()/(2*d)
    analytic=th[:,2]+th[:,3]*basis(t(L),th[:,0],True)
    energy=float(((z*z).sum(1)-(QVIN[0]**2+QVIN[1]**2)).max())
    assert energy<1e-12 and float(abs(numeric-analytic).max())<1e-8
    # Factorization should retain both marginals and sum to one.
    w=b.posterior(json.loads((R/'pilot/public/0000.json').read_text()));f=b.factorize(w)
    q=w.reshape(2,2,21,21,21);qq=f.reshape(q.shape)
    marginal=max(float(abs(q.sum((1,4))-qq.sum((1,4))).max()),float(abs(q.sum((0,2,3))-qq.sum((0,2,3))).max()))
    assert marginal<1e-12
    write(R/'validation.json',dict(energy_excess=energy,slope_fd_max=float(abs(numeric-analytic).max()),factorized_marginal_error=marginal))

def acquire(split,n=31,H=256):
    b=Bank(n);count=SPLITS[split][0]
    for i in range(count):
        ppath=R/split/'public'/f'{i:04d}.json';p=json.loads(ppath.read_text());start=time.time()
        w=b.posterior(p);outer=b.outer(w,H,120000+i);jr,pr,ess=b.risks(w,outer);mr=b.last_mixture_risk.copy()
        wf=b.factorize(w);fo=b.outer(wf,H,120000+i);fr,_,_=b.risks(wf,fo)
        sr,cr,se=b.snis(w,outer,130000+i)
        actions=dict(joint=int(jr.argmin()),plugin_aware=int(pr.argmin()),mixture_aware=int(mr.argmin()),factorized=int(fr.argmin()),standard_snis=int(sr.argmax()),cf_snis=int(cr.argmax()),none=0)
        out=dict(actions=actions,joint_risk=jr.tolist(),plugin_risk=pr.tolist(),mixture_risk=mr.tolist(),factorized_risk=fr.tolist(),snis_utility=sr.tolist(),cf_utility=cr.tolist(),ess=ess,snis_ess=se,diagnostics=b.diagnostics(w),seconds=time.time()-start,public_sha256=sha(ppath),order=n,H=H)
        write(R/split/'policies'/f'{i:04d}.json',out)
        if i%32==0:print('acquire',split,i,out['actions'],out['seconds'],flush=True)
    del b;torch.cuda.empty_cache()

def freeze(split):
    write(R/split/'freeze.json',dict(policies={p.name:sha(p) for p in sorted((R/split/'policies').glob('*.json'))},public={p.name:sha(p) for p in sorted((R/split/'public').glob('*.json'))},rules_sha256=sha(R/'rules.npz'),time=time.time()))

def evaluate(split,n=41):
    b=Bank(n);fz=json.loads((R/split/'freeze.json').read_text());assert sha(R/'rules.npz')==fz['rules_sha256']
    out=R/split/f'evaluation_{n}';out.mkdir(parents=True,exist_ok=True)
    for i in range(SPLITS[split][0]):
        name=f'{i:04d}.json';pp=R/split/'public'/name;policy=R/split/'policies'/name
        assert sha(pp)==fz['public'][name] and sha(policy)==fz['policies'][name]
        p=json.loads(pp.read_text());w=b.posterior(p);wf=b.factorize(w)
        # Private outcomes are accessed only after all policies and rules are frozen.
        truth=np.load(R/split/'private'/f'{i:04d}.npz');pred={k:[] for k in ['joint','plugin','mixture','factorized']}
        for a in range(4):
            ww=w if a==0 else b.update(w,a,truth['obs'+str(a)])
            ff=wf if a==0 else b.update(wf,a,truth['obs'+str(a)])
            mm,pl,mix=b.decode(ww,True);fm,_=b.decode(ff)
            for k,v in [('joint',mm),('plugin',pl),('mixture',mix),('factorized',fm)]:pred[k].append(v.expand(32,-1).cpu().numpy())
        output={k+'_prediction':np.stack(v) for k,v in pred.items()}
        output.update({k+'_loss':.5*((np.stack(v)-truth['target'])**2).sum(-1) for k,v in pred.items()})
        output['target']=truth['target'];np.savez_compressed(out/f'{i:04d}.npz',**output)
        if i%64==0:print('evaluate',split,n,i,flush=True)
    del b;torch.cuda.empty_cache()

def audit(n=61,H=1024,count=32):
    b=Bank(n);records=[]
    for i in range(count):
        p=json.loads((R/'test/public'/f'{i:04d}.json').read_text());w=b.posterior(p)
        start=time.time();reps=[]
        for seed in [410000+i,510000+i]:
            outer=b.outer(w,H,seed);jr,pr,_=b.risks(w,outer);reps.append(dict(risk=jr.tolist(),plugin_risk=pr.tolist()))
        records.append(dict(world=i,repeats=reps,seconds=time.time()-start))
        write(R/'ranking_audit.json',records)
        if i%8==0:print('audit',i,flush=True)

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'Run on allocated compute node'
    torch.set_num_threads(2);cmd=sys.argv[1]
    if cmd=='prepare':prepare()
    elif cmd=='pilot':pilot()
    elif cmd=='acquire':acquire(sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 31)
    elif cmd=='freeze':freeze(sys.argv[2])
    elif cmd=='evaluate':evaluate(sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 41)
    elif cmd=='audit':audit()
