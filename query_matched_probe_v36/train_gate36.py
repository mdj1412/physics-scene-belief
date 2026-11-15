"""Build v36 gate exclusively from v33/v34 source laws and archived Brier targets."""
from pathlib import Path
import hashlib
import json
import numpy as np
from gate36 import ROOT,BETA,FY,MODES,FAMILIES,SIGNS,observations

SOURCE=ROOT.parent/'bidirectional_probe_v34'
ZGRID=np.linspace(-1.,1.,41)
STRENGTH=(1.,.30,0.,0.)
SOURCE_SEED=360021776
N_COMPONENT=64
N_PER_COMP=384

def parameters(u,flip):
    wcurve=(u[:,0]>=.5).astype(float); wmu=(u[:,1]>=.5).astype(float)
    csgn=2*(u[:,2]>=.5).astype(float)-1; msgn=2*(u[:,3]>=.5).astype(float)-1
    offset=.00115*(2*wcurve-1)*(1-2*flip[:,0])+.0034*(u[:,6]-.5)
    slope=.070*(u[:,7]-.5)
    curve=csgn*(.022+.318*wcurve)+.024*(u[:,4]-.5)
    mu=.251+msgn*(.011+.127*wmu)+.015*(u[:,5]-.5)
    restitution=.70+.095*(2*wmu-1)*(1-2*flip[:,1])+.18*(u[:,8]-.5)
    return np.column_stack([offset,slope,curve,mu,restitution])

def shape(theta,family):
    th=theta.copy()
    th[:,2]*=.90 if family.startswith('rounded') else 1.10
    th[:,3]*=.98 if family.endswith('fine') else 1.04
    return th

def impact(theta,height,speed,family,sign,strength):
    _,s,c,mu,e=theta.T
    geom=-.060 if family.startswith('rounded') else .060
    amp,tau=(.17,.34) if family.endswith('fine') else (.22,.50)
    ss=s+c*(height/.025)+geom*(height/.025)**2
    norm=np.sqrt(1+ss**2)
    vn=(1-ss*speed)/norm
    vt=(speed+ss)/norm
    jn=(1+e)*np.maximum(vn,0)
    capacity=jn*mu*(1+sign*strength*amp*np.tanh(vt/tau))
    stick=abs(vt)<=capacity
    jt=np.where(stick,vt,np.sign(vt)*capacity)
    velocity=np.column_stack((1-(jn+jt*ss)/norm,speed+(jn*ss-jt)/norm))
    return velocity,stick

def factual(theta,family):
    off,s,c,_,_=theta.T
    geom=-.060 if family.startswith('rounded') else .060
    curve=off[:,None]+s[:,None]*FY+c[:,None]*FY**2/.05+geom*FY**3/(3*.025**2)
    velocity,stick=impact(theta,0,0,family,1,0)
    assert stick.all()
    return np.column_stack((curve,velocity))

def mixture_samples():
    rng=np.random.default_rng(SOURCE_SEED)
    group=np.arange(N_COMPONENT).repeat(N_PER_COMP)
    bits=((group[:,None]>>(np.arange(6)[None,:]))&1)
    unit=rng.uniform(size=(len(group),9))
    unit[:,:4]=(unit[:,:4]+bits[:,:4])/2.
    # Conditional proxy flips are marginalized. Source alpha varies across environments.
    return parameters(unit,bits[:,4:]).reshape(N_COMPONENT,N_PER_COMP,5)

def components(th,family,sign,strength,height,incoming):
    th=th.reshape(-1,5)
    vel,_=impact(th,height,incoming,family,sign,strength)
    points=observations(factual(th,family),vel).reshape(N_COMPONENT,N_PER_COMP,7)
    center=points.mean(1)
    residual=points-center[:,None]
    # Each projected observed axis has independent unit sensor noise.
    covariance=np.einsum('kni,knj->kij',residual,residual)/(N_PER_COMP-1)+np.eye(7)
    precision=np.linalg.inv(covariance)
    log_norm=-.5*np.linalg.slogdet(covariance)[1]-np.log(N_COMPONENT)
    return center,precision,log_norm

def source_truth(seed,alpha,family,sgn):
    # Exact NumPy version of v27's draw from the archived source deployment seed and lane.
    u=np.random.default_rng(np.random.SeedSequence([seed,270000,20])).random((128,9))
    flips=np.random.default_rng(np.random.SeedSequence([seed,270530])).random((128,2)) < ((1.-alpha)/2.)
    th=shape(parameters(u,flips.astype(int)),family)
    return [impact(th,.025,.425,family,sgn,r)[1] for r in STRENGTH]

def gain_curve(family,sgn):
    result=np.zeros((len(MODES),len(ZGRID)))
    for b in range(8):
        ori='plus' if sgn==1 else 'minus'
        folder=SOURCE/'results_policy'/f'{family}_{ori}'/f'block_{b:03d}'/'alternating'
        meta=json.loads((folder/'meta.json').read_text())
        with np.load(folder/'metrics.npz') as pack:
            choices=(pack['actions'][:,:,1]==3).astype(int)
            p=np.take_along_axis(pack['probabilities'],choices[:,:,None,None],axis=2)[:,:,0]
            yy=np.array([source_truth(sd,al,family,sgn) for sd,al in zip(meta['deployment_seeds'],pack['alpha'])]).transpose(1,0,2)
            np.testing.assert_array_equal(yy[0],pack['labels'])
            np.testing.assert_array_equal(yy[-1],pack['nominal_labels'])
            for i,z in enumerate(ZGRID):
                odds=np.exp(BETA*z)
                d=p*odds/(1-p+p*odds)-p
                penalty=np.mean(d*d+2*d*p)
                result[:,i]+=2*np.mean(yy*d.mean(-1)[None],axis=(1,2))-penalty
    return result/8.

def train():
    theta=mixture_samples()
    samples=[]; curves=[];signs=[];far=[];near=[]
    for family in FAMILIES:
        th=shape(theta.reshape(-1,5),family)
        for sign in SIGNS:
            gain=gain_curve(family,sign)
            for mode,scale in enumerate(STRENGTH):
                far_scale=float(mode!=3)
                arr=[components(th,family,sign,far_scale,0,spd) for spd in (-1.10,1.10)]
                arr.append(components(th,family,sign,scale,.023,.50))
                samples.append(arr);curves.append(gain[mode]);signs.append(sign)
                far.append(sign*far_scale);near.append(scale)
    arrays={field:np.stack([np.stack([row[k] for row in hypothesis]) for hypothesis in samples],axis=1)
            for k,field in enumerate(['means','precision','log_norm'])}
    np.savez_compressed(ROOT/'gate36_frozen.npz',**arrays,sign=signs,far_sign=far,near_strength=near,
                        gain=curves,zgrid=ZGRID)
    filehash=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    audit=dict(artifact_sha256=filehash(ROOT/'gate36_frozen.npz'),source_seed=SOURCE_SEED,source_families=FAMILIES,
               components=N_COMPONENT,samples_per_component=N_PER_COMP,source_archives=64,
               source_reference='v34 fixed4 alternating selected deployment action from 4 families x 2 signs x 8 blocks',
               source_label_replay='all original strong labels and isotropic nominal labels matched exactly',
               calibration_height=.023,calibration_speed=.50,beta=BETA,
               training_code_sha256=filehash(Path(__file__)),inference_code_sha256=filehash(ROOT/'gate36.py'))
    (ROOT/'gate36_freeze.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(audit)
if __name__=='__main__':train()
