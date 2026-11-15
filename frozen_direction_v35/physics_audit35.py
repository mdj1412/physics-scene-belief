"""Independent NumPy verification of simulator truth, transport and provenance."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from audit35 import OBJECTS, SURFACES, SIGNS, SCHEDULES, ROOT, SOURCE

SOURCE_SCALE = {'fine':(.17,.34),'coarse':(.22,.50)}
NEW_SURFACES = SURFACES+('isotropic',)


def perturb(u,obj):
    if obj=='sinusoid':
        return .045*.025*(np.cos(u)-1)
    return .040*.025*u**4/4


def slope_and_coefficient(theta, obj, surf, sign, height, incoming):
    u = height/.025
    s = theta[...,2]+theta[...,3]*u
    s += -.045*np.sin(u) if obj=='sinusoid' else .040*u**3
    normal = np.sqrt(s*s+1)
    vt = (incoming+s)/normal
    frac = {'atan':.21*2/np.pi*np.arctan(vt/.40),
            'algebraic':.16*vt/np.sqrt(.30**2+vt**2),'isotropic':np.zeros_like(vt)}[surf]
    mu=theta[...,4]*(1+sign*frac)
    n=(1+theta[...,5])*np.maximum((1-s*incoming)/normal,0)
    return s,vt,mu,n


def verify(root):
    result={}
    hashes=json.loads((root/'run.json').read_text())['hashes']
    for name,actual in hashes.items():
        assert hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()==actual, name
    for o in OBJECTS:
        yy=np.linspace(-.025,.025,101)
        eps=1e-7
        numer=(perturb((yy+eps)/.025,o)-perturb((yy-eps)/.025,o))/(2*eps)
        formula = -.045*np.sin(yy/.025) if o=='sinusoid' else .040*(yy/.025)**3
        np.testing.assert_allclose(numer,formula,atol=1e-10,rtol=1e-8)
    n=0
    seen=set()
    for obj,surf in itertools.product(OBJECTS,NEW_SURFACES):
        obs=[]
        for b in range(8):
            twins={}
            for sgn in (SIGNS if surf!='isotropic' else ('plus',)):
                folder=root/f'{obj}_{surf}_{sgn}'/f'block_{b:03d}'
                meta=json.loads((folder/'meta.json').read_text())
                orient=1 if sgn=='plus' else -1
                assert meta['sign']==orient and meta['object']==obj and meta['surface']==surf
                seedset=set(meta['calibration_seeds']+meta['deployment_seeds']+meta['pair'])
                assert len(seedset)==18
                if sgn=='plus':
                    assert seedset.isdisjoint(seen);seen.update(seedset)
                else:
                    assert seedset==save_seeds
                save_seeds=seedset
                for sch in SCHEDULES:
                    path=folder/(sch+'.npz')
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['archives'][sch]
                    n+=1
                    with np.load(path) as fp:
                        h=np.array(fp['theta']); y=fp['labels'].astype(bool)
                        ids=fp['paid_ids']
                        assert ids.shape==(8,4) and all(len(set(ii))==4 for ii in ids)
                        assert ((fp['actions']==3).sum(axis=1)==13).all()
                        np.testing.assert_array_equal(fp['betas'],np.array(json.loads((SOURCE/'anchored34.json').read_text())['beta'])[:,0])
                        if sch=='alternating':
                            np.testing.assert_array_equal(h,h0);np.testing.assert_array_equal(y,y0)
                            np.testing.assert_array_equal(ids,id0)
                            continue
                        h0,y0,id0=h,y,ids
                        s,vt,mu,jn=slope_and_coefficient(h,obj,surf,orient,.025,.425)
                        assert np.all(mu>0) and np.all(jn>0)
                        np.testing.assert_array_equal(np.abs(vt)<=mu*jn,y)
                        for incoming,expect_stick in [(0,True),(-1.10,False),(1.10,False)]:
                            _,vtc,mc,nc=slope_and_coefficient(h,obj,surf,orient,0,incoming)
                            assert np.all((np.abs(vtc)<=mc*nc)==expect_stick)
                        twins[sgn]=(h,y,id0)
                        obs.append([mu.min(),mu.max(),vt.min(),vt.max(),((vt<0)).mean(),y.mean()])
            if surf!='isotropic':
                hp,yp,idp=twins['plus'];hm,ym,idm=twins['minus']
                np.testing.assert_array_equal(hp,hm);np.testing.assert_array_equal(idp,idm)
                assert (yp>=ym).all()
        v=np.array(obs)
        result[obj+'_'+surf]=dict(mu_min=v[:,0].min(),mu_max=v[:,1].max(),query_vt_min=v[:,2].min(),
            query_vt_max=v[:,3].max(),negative_query_vt_fraction=v[:,4].mean(),stick_rate=v[:,5].mean())
    u=np.array([.1,.425,1.1])
    curves={name:(amp*np.tanh(u/tau)).tolist() for name,(amp,tau) in SOURCE_SCALE.items()}
    curves.update(atan=(.21*(2/np.pi)*np.arctan(u/.40)).tolist(),algebraic=(.16*u/np.sqrt(.30**2+u**2)).tolist())
    return dict(archives_verified=n,input_files_verified=len(hashes),unique_source_seeds=len(seen),
                cases=result,relative_friction_modulation_at_vt_0p1_0p425_1p1=curves)


if __name__=='__main__':
    out=verify(ROOT/'results')
    (ROOT/'physics_audit35.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
