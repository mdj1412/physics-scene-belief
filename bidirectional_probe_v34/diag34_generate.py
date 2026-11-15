"""Generate independent paired signed environments on a Slurm compute CPU."""
import argparse,os,sys,json,hashlib,socket,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'directional_prefix_v33'))
from target34 import directions,calibration
from target33 import draw_object,contact,contour,FY,FACTUAL_INCOMING,PROBE_INCOMING,sigmas,stream,torch,t,P
from diag34_model import raw_features,fit,predict
FAMILIES=('rounded_fine','rounded_coarse','ridged_fine','ridged_coarse')
SCHED=('same','alternating')
STAGES=(2,4,8)
OUT=ROOT/'results_diag'
config=json.loads((ROOT/'policy_design34.json').read_text())
ALPHAS=config['alphas'] if 'alphas' in config else json.loads((ROOT.parent/'directional_prefix_v33/design_eval.json').read_text())['alphas']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def block(fam_index,b,split,detector):
    fam=FAMILIES[fam_index]
    prefix_seed=1034000000+1000000*fam_index+{'train':0,'test':400000,'scalar_test':800000}[split]+b*100
    xpaid=[];ys=[];ds=[];couplings=[];summaries=[];scalars=[];seeds=[]
    model=json.loads((ROOT.parent/'compute_matched_stopping_v32/results_train/model_lock.json').read_text())['models']['broad']
    from score30 import features,predict as frozen_score
    alphas=np.asarray(ALPHAS)[stream(prefix_seed,32).permutation(8)]
    for e,alpha in enumerate(alphas):
        seed=prefix_seed+e; start=(-1,1)[(b+e)%2];seeds.append(seed)
        lplus=fam+'_plus'
        theta=draw_object(1024,seed,10,alpha,lplus)
        f,mask=contact(theta,0,FACTUAL_INCOMING,lplus)
        assert bool(mask.all())
        f0=torch.cat((contour(theta,FY,lplus),f),-1)
        x=(f0+t(stream(seed,11).normal(size=(1024,11)))*sigmas()[0]).cpu().numpy()
        ids=stream(seed,98).permutation(len(x))[:8]
        epsi=stream(seed,99).normal(size=(8,2))*sigmas()[P].cpu().numpy()
        raw_y=np.empty((2,2,8,2)); dirs=np.stack([directions(s,start) for s in SCHED])
        logf=detector.component_loglikelihood(x) if detector else None
        ff=np.zeros((2,2,2,15));ss=np.zeros((2,2,2))
        for o,tag in enumerate(('plus','minus')):
            law=fam+'_'+tag
            for j,schedule in enumerate(SCHED):
                di=dirs[j]
                measured,sticks=contact(theta[ids],0,t(di)*abs(PROBE_INCOMING),law)
                assert not bool(sticks.any())
                y=measured.cpu().numpy()+epsi
                raw_y[o,j]=y
                if b==0 and e==0:
                    xx,ii,yy,dd=calibration(seed,alpha,law,schedule,start)
                    np.testing.assert_array_equal(xx,x);np.testing.assert_array_equal(yy,y)
                    np.testing.assert_array_equal(ii,ids);np.testing.assert_array_equal(dd,di)
                if detector:
                    joint=detector.signed_joint(x[ids],y,di)
                    mapped=y.copy();mapped[:,1]*=-di
                    for si,k in enumerate(STAGES[:2]):
                        post=detector.posterior_from_components(logf,ids[:k],joint[:k])
                        ff[o,j,si]=features(post,x,ids[:k],mapped[:k])
                        ss[o,j,si]=frozen_score(model[si],ff[o,j,si])
        xpaid.append(x[ids]);ys.append(raw_y);ds.append(dirs);couplings.append(alpha);summaries.append(ff);scalars.append(ss)
    name=OUT/f'{split}_{fam}_{b:03d}.npz' if not detector else OUT/f'heads_{split}_{fam}_{b:03d}.npz'
    np.savez_compressed(name,x=np.asarray(xpaid),y=np.array(ys),d=np.array(ds),alpha=np.array(couplings),
                        features=np.array(summaries),scores=np.array(scalars),seeds=np.array(seeds),has_heads=detector is not None)


def direct_data(fam,split,stage,j,small=False):
    size=(8 if small else (64 if split=='train' else 48))
    feature=[];target=[]
    for b in range(size):
        with np.load(OUT/f'{split}_{fam}_{b:03d}.npz') as z:
            for o in range(2):
                feature.extend(raw_features(z['x'][:,:stage],z['y'][:,o,j,:stage],z['d'][:,j,:stage]))
                target.extend(np.full(8,1-o))
    return np.array(feature),np.array(target)


def lock_models():
    models={}
    for i,fam in enumerate(FAMILIES):
        other=FAMILIES[3-i]; models[fam]={}
        for j,s in enumerate(SCHED):
            models[fam][s]={}
            for k in STAGES:
                a,b=direct_data(other,'train',k,j)
                models[fam][s][str(k)]=fit(a,b)
    out=dict(models=models,input='no simulator state: only paid x,y,direction',family_order=FAMILIES,training_hashes={
       f.name:sha(f) for f in sorted(OUT.glob('train_*.npz'))},
       plan_sha256=sha(ROOT/'DIAG_PLAN_KO.md'),readout_sha256=sha(ROOT/'diag34_model.py'))
    dest=ROOT/'diag_models.json';temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(out,indent=2)+'\n');temp.replace(dest)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('raw','heads'));m=parser.parse_args().mode
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    detector=None
    if m=='heads':
        from detector34 import ActionConditionedDetector
        detector=ActionConditionedDetector(seed=3003913,bank_size=8192)
    started=time.time()
    with torch.inference_mode():
        for split,n in [('train',64),('test',48)]:
            for i,fam in enumerate(FAMILIES):
                for b in range(8 if detector else n):
                    block(i,b,split,detector)
                print(m,split,fam,round(time.time()-started,1),flush=True)
            if split=='train' and m=='raw':lock_models()
    spec=dict(mode=m,job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
              elapsed_seconds=time.time()-started,plan_sha256=sha(ROOT/'DIAG_PLAN_KO.md'))
    (OUT/(m+'_complete.json')).write_text(json.dumps(spec,indent=2)+'\n')
if __name__=='__main__':main()
