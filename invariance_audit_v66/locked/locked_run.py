"""Generate observations, freeze predictions, then score a separately released solver."""
import argparse, hashlib, itertools, json, os, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJ=ROOT.parent.parent
sys.path[:0]=[str(PROJ/'normal_variance_v46'),str(PROJ/'image_pressure_v45'),str(PROJ/'contact_patch_v44')]
import camera46, camera45, runner44 as old, pressure45
from locked_geometry import FAMILIES, VARIANTS, profile
r,infer,torch=old.r,old.infer,old.torch
HEIGHTS=(19,31)
CELLS=tuple(itertools.product(FAMILIES,pressure45.SURFACES,pressure45.LAWS,VARIANTS,HEIGHTS))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(name, data): (ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def configure():
    old.BASE,old.SUPPORT_BASE,old.OUT=660900000,980,ROOT/'results'
    old.CELLS=tuple((*cell[:-1],cell[-1]/1000.) for cell in CELLS)
    old.world.TARGET_SHAPES=FAMILIES
    def pert(y,shape,var):
        y=old.world.t(y)
        f,g=profile(y.detach().cpu().numpy()*1000,shape,var)
        return tuple(torch.as_tensor(a,device=y.device,dtype=y.dtype) for a in (g,.001*f))
    old.world.perturbation=pert
    camera46.shape_function=lambda family:profile
    pressure45.ROOT=ROOT
    pressure45.slope=lambda x,shape,var:profile(x,shape,var)[1]
    pressure45.output_folder=lambda high:ROOT/'labels'
def world():
    for panel in range(12):old.manifest(panel)
    x=np.linspace(-10,46,1401);eps=2e-6
    derivative={}
    for f,v in itertools.product(FAMILIES,VARIANTS):
        y,g=profile(x,f,v)
        fd=(profile(x+eps,f,v)[0]-profile(x-eps,f,v)[0])/(2*eps)
        error=float(np.max(abs(g-fd)))
        assert error<2e-8 and np.max(np.abs(profile(0,f,v)))<1e-12
        derivative[f+'_'+v]=error
    # Test actual six-parameter prototypes against every located pre-existing catalog manifest.
    def keys(path):
        with np.load(path,allow_pickle=False) as f:
            if 'theta' not in f.files:return set()
            theta=f['theta']
        if theta.shape[-1]!=6:return set()
        return set(map(tuple,theta.reshape(-1,6)))
    previous=set(); earlier=sorted(PROJ.glob('*/results/worlds/panel_*.npz'))
    for p in earlier:previous.update(keys(p))
    new=set()
    for pan in range(12):new.update(keys(ROOT/'results/worlds'/f'panel_{pan:02d}.npz'))
    overlap=len(new&previous)
    assert overlap==0
    dump('world_audit.json',dict(max_derivative_error=derivative,new_distinct=len(new),earlier_files=len(earlier),earlier_distinct=len(previous),intersection=overlap))
def vision():
    for panel in range(12):
        sub=f'panel_{panel:02d}'
        with np.load(ROOT/'results/worlds'/(sub+'.npz')) as z:theta=z['theta'].reshape(512,6)
        out=ROOT/'routes'/sub;out.mkdir(parents=True,exist_ok=True)
        for f,v,h in itertools.product(FAMILIES,VARIANTS,HEIGHTS):
            path=out/f'{f}_{v}_h{h}.npz'
            x=np.empty((3,512),np.float32);dig=hashlib.sha256()
            for off,d in enumerate(pressure45.OFFSETS):
                for k in range(0,512,128):
                    fr=camera46.render_clip(theta[k:k+128],f,v,h,d,66032026+128*panel+k//128)
                    dig.update(fr.numpy().tobytes())
                    x[off,k:k+128]=camera45.infer_x(fr).numpy()
            x=x.reshape(3,4,128)
            old.v38.atomic_npz(path,x_hat_mm=x,route=camera45.source_gate(torch.from_numpy(x)).numpy(),raw_clip_hash=dig.hexdigest())
        print('vision',panel,flush=True)
def predict():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    engine=infer.make_bank();axes=json.loads((PROJ/'image_pressure_v45/source_axis45.json').read_text())
    templates={}
    for h in HEIGHTS:
        templ=infer.make_template(engine,h/1000.)
        c,n,l=templ
        u=r.v38.t(axes[str(h)]['axis'])
        cnew=c-(1-.25**2)*(c@u)[:,None]*u
        templates[h]=[templ,infer.make_template(engine,h/1000.,axes[str(h)]['axis']),(cnew,.5*(c*cnew).sum(-1),l)]
    for panel in range(12):
        cache={}
        for f,s,law,v,h in CELLS:
            cell=f,s,law,v,h/1000
            path=old.path_for(panel,cell)
            th,xx,obs,_=old.observations(panel,cell)
            key=hashlib.sha256(xx.tobytes()).digest()
            if key not in cache:cache[key]=infer.actions43(xx,engine)
            act=cache[key];paid=infer.paid_sensors(obs,act)
            pred=np.stack([infer.decode43(engine,templ,xx,act,paid) for templ in templates[h]])[:,None]
            assert pred.shape==(3,1,4,128,4) and np.isfinite(pred).all() and pred.min()>=-1e-12 and pred.max()<=1+1e-12
            for j in (1,2):np.testing.assert_allclose(pred[j,0][act==r.P],pred[0,0][act==r.P],atol=1e-13)
            path.parent.mkdir(parents=True,exist_ok=True)
            old.v38.atomic_npz(path,theta=th,xx=xx,selected=act,query_paid=np.stack([s[r.Q] for s in paid]),probe_paid=np.stack([s[r.P] for s in paid]),pred=pred,decoders=['original','source_axis','soft075'])
        print('predict',panel,flush=True)
def primary():
    assert (ROOT/'FORECASTS_SEALED').is_file()
    for p,f,v,h in itertools.product(range(12),FAMILIES,VARIANTS,HEIGHTS):
        pressure45.run_group(p,f,v,h,False)
    print('576 primary label archives',flush=True)
def score():
    from scipy.stats import t
    def ci(x,n=1):
        m=x.mean(0);hw=t.isf(.025/n,11)*x.std(0,ddof=1)/np.sqrt(12)
        return np.stack((m,m-hw,m+hw),-1).tolist()
    scores=np.zeros((12,2,3))
    for p in range(12):
        sub=f'panel_{p:02d}'
        for f,s,law,v,h in CELLS:
            name=f'{f}_{s}_{law}_{v}_h{h}.npz'
            with np.load(ROOT/'results'/sub/name) as z:pred=z['pred'][:,0]
            with np.load(ROOT/'labels'/sub/name) as z:y=z['labels'][1:]
            with np.load(ROOT/'routes'/sub/f'{f}_{v}_h{h}.npz') as z:gate=z['route']
            both=np.where(gate[None,None,...,None],pred[:,None,None],pred[0][None,None,None])
            loss=((both-y[None,...,None])**2).mean((1,2,3,4,5))
            scores[p,FAMILIES.index(f)]+=loss/24
    pooled=scores.mean(1);contrasts=pooled[:,2,None]-pooled[:,[0,1]]
    dump('locked_results.json',dict(order=['full','hard','soft075'],means=pooled.mean(0).tolist(),comparisons=ci(contrasts,2),family_mean=scores.mean(0).tolist(),positive_per_comparison=(contrasts>0).sum(0).tolist(),panel_family=scores.tolist(),independent_catalogs=12))
def main():
    assert os.getenv('SLURM_JOB_ID')
    torch.set_num_threads(1);configure()
    mode=argparse.ArgumentParser();mode.add_argument('phase',choices=['world','vision','predict','primary','score'])
    with torch.inference_mode():globals()[mode.parse_args().phase]()
if __name__=='__main__':main()
