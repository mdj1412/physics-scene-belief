"""Replay the locked source contour forecast on disjoint v39 object and noise indices."""
import json
from pathlib import Path
import numpy as np
import forecast40 as frozen

ROOT=Path(__file__).resolve().parent

def replication(path,shape,old_law,catalog):
    meta=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as f:
        alpha,z,w,y,a,p=[f[k] for k in ('alpha','z','weights','labels','actions','p_base')]
    picks=np.arange(2*meta['block']+1,128,4)
    noise=np.array([8,24])
    shifts=(-frozen.BUMP,frozen.BUMP)
    result=np.zeros((2,2))
    for e,aa in enumerate(alpha):
        kernel=frozen.policy.Engine(meta['pair'],*aa).inner
        for s,sign in enumerate((1,-1)):
            obs,_,_,sensor=frozen.historical.deployment(meta['deployment_seeds'][e],aa,
                        (shape,old_law,sign,catalog),'strong',repeats=32)
            ids=picks[a[s,e,picks]==frozen.policy.Q]
            x=sensor[frozen.policy.Q][ids][:,noise]
            prior=kernel.fit(frozen.policy.t(obs[ids]))[:,None]
            inputs=np.stack([x,x+shifts[0],x+shifts[1]],axis=1)
            post=kernel.update(prior[:,None],frozen.policy.Q,frozen.policy.t(inputs))
            probabilities=(post@kernel.label[:,0]).numpy()/np.sqrt(2.)
            np.testing.assert_allclose(probabilities[:,0],p[s,e,ids][:,noise],rtol=0,atol=2e-10)
            for r in range(3):
                def correct(weight):
                    odds=np.exp(frozen.BETA*z[s,e]*weight)
                    return probabilities*odds/(1-probabilities+probabilities*odds)
                fixed,opt=correct(.325),correct(w[s,r,e])
                for j,law in enumerate(frozen.LAWS):
                    rho=frozen.retention(old_law,law)
                    mix=.5*(1+rho)*y[s,r,e,ids]+.5*(1-rho)*y[1-s,r,e,ids]
                    lossdiff=(opt-fixed)*(opt+fixed-2*mix[:,None,None])
                    result[:,j]+=(lossdiff[:,1:]-lossdiff[:,:1]).sum((0,2))/(len(noise)*len(picks))
    return result/(8*2*3)

def main():
    frozen.policy.torch.set_num_threads(1)
    table=json.loads((ROOT/'forecast40.json').read_text())['cells']
    item={}
    for family in frozen.SHAPES:
        effects=[]
        for old in frozen.OLD_LAWS:
            for k in range(12):
                for b in range(2):
                    src=frozen.OLD/'results'/f'{family}_{old}'/f'catalog_{k:02d}'/f'block_{b:02d}.npz'
                    effects.append(replication(src,family,old,k))
        average=np.mean(effects,axis=0)
        for i,variant in enumerate(('hollow','bulge')):
            for j,law in enumerate(frozen.LAWS):
                key=f'{family}_{variant}_{law}'
                expected=table[key]
                shift=float(average[i,j])
                alternate=expected['law_replay_no_contour']+shift
                item[key]=dict(frozen_delta=expected['delta'],disjoint_delta=alternate,
                    frozen_contour_difference=expected['geometry_increment'],disjoint_contour_difference=shift)
    (ROOT/'source_contour_check40.json').write_text(json.dumps(item,indent=2)+'\n')
if __name__=='__main__':main()
