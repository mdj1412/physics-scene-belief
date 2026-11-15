"""Independently reconstruct every deployed decision and Brier risk from raw evidence."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'probe_speed_v37'))
from gate37 import load_artifacts
from gate36 import observations, log_density, predict

METHODS = ('fixed','binary','ternary','continuous')
ACTIONS = np.array([.325, 0., 1.])
BETA = .8289760825845313


def source_weights(art,x,ys,dirs,y4):
    n = len(x)
    check = predict(art,x,ys,dirs,y4,'alternating')
    # Separate reconstruction of the v37 diagnostic. Interpolate via np.interp
    # rather than the online routines' index arithmetic and weighted sum.
    near = log_density(art,observations(x[:,3],y4),2)
    far = np.zeros_like(near)
    for i in range(n):
        for j in range(3):
            inp = observations(x[i:i+1,j],ys[i:i+1,j])
            far[i] += log_density(art,inp,int(dirs[i,j]>0))[0]
    out = np.empty((2,4,n))
    for k, ll in enumerate((far,far+near)):
        w = np.exp(ll-logsumexp(ll,axis=1)[:,None])
        for i,z in enumerate(check['z']):
            gains = np.array([[np.interp(f*z,art['zgrid'],row) for row in art['gain']]
                              for f in (.325,1.)]) @ w[i]
            scores = [gains[0],0.,gains[1]]
            out[k,:,i] = [.325,float(check[f'accept{k+3}'][i]),ACTIONS[np.argmax(scores)],
                          check[f'near_strength{k+3}'][i]]
    return check['z'],out


def audit_legacy(art):
    folder=ROOT.parent/'probe_speed_v37/results'
    scopes=json.loads((ROOT.parent/'probe_speed_v37/ternary37.json').read_text())['scopes']['local_equal']
    rows=[]
    for f in sorted(folder.glob('*/block_*/*.npz')):
        if 'global_null' in f.name:
            continue
        with np.load(f) as a:
            z,w=source_weights(art,a['far_x'],a['first3_y'],a['dirs'],a['fourth_y'][2])
            p=a['p_base'][2]
            odds=np.exp(BETA*z*w[1,2])[:,None,None]
            prob=p*odds/(1-p+p*odds)
            rows.append(np.square(prob-a['labels'][:,:,None]).mean())
    if not rows:
        raise ValueError('no legacy archives')
    val=float(np.mean(rows))
    np.testing.assert_allclose(val,scopes['means'][2][0],atol=1e-12,rtol=0)
    return dict(n=len(rows),Brier=val)


def main():
    art=load_artifacts()[1.1]
    if '--v37' in sys.argv:
        print(json.dumps(audit_legacy(art)))
        return
    out=ROOT/'results'
    done=json.loads((out/'completion.json').read_text())
    blocks=sorted(out.glob('*/block_*/meta.json'))
    assert len(blocks)==done['blocks']==32
    max_risk_error=0.
    lower, upper, above_one = 1., 0., 0
    clusters=0
    count=0
    seeds=[]
    # Paired empirical identity. Each term includes the frequency of the choice.
    # Denominator is all environment rows, never the selected subset.
    transitions = np.empty((4,8,3,2,3,3,2))
    chosen_terms = np.empty((4,8,3,2,3,2))
    tiers = ('strong','weak','null')
    cells = ('shoulder_weakening','shoulder_strengthening','bead_weakening','bead_strengthening')
    labels_action = np.array([0.,.325,1.])
    for m in blocks:
        metadata=json.loads(m.read_text())
        b=m.parent
        seeds.extend(metadata['calibration_seeds']+metadata['deployment_seeds'])
        assert len(metadata['sha256'])==8
        for path,expected in metadata['sha256'].items():
            h=hashlib.sha256((b/path).read_bytes()).hexdigest()
            assert h==expected
        for sign in ('plus','minus'):
            with np.load(b/f'shared_{sign}.npz') as x:
                actions=x['actions']; n,objects=actions.shape
                assert (n,objects)==(8,128)
                assert np.all((actions==int(x['P'])).sum(1)==13)
                assert np.all((actions==int(x['Q'])).sum(1)==115)
                assert x['probe_obs'].shape[:3]==(8,13,32) and x['query_obs'].shape[:3]==(8,115,32)
                p=x['p_base']
                assert p.shape==(8,128,32) and np.isfinite(p).all() and np.all((p >= -1e-14) & (p <= 1+1e-14))
                lower,upper=min(lower,float(p.min())),max(upper,float(p.max()))
                above_one+=int((p>1).sum())
                for regime in ('strong','weak','null'):
                    with np.load(b/f'{regime}_{sign}.npz') as a:
                        assert tuple(a['methods'])==METHODS
                        np.testing.assert_array_equal(a['stages'],[3,4])
                        z,w=source_weights(art,x['far_x'],x['first3_y'],x['dirs'],a['fourth_y'])
                        np.testing.assert_allclose(z,x['z'],atol=1e-12,rtol=0.)
                        np.testing.assert_allclose(w,a['weights'],atol=1e-12,rtol=0.)
                        assert a['labels'].dtype==bool
                        odds=np.exp(BETA*w*z)[:,:,:,None,None]
                        probs=p*odds/(1-p+p*odds)
                        risk=((probs-a['labels'][:,:,None])**2).mean(axis=(-2,-1))
                        diff=np.abs(risk-a['risk']).max()
                        max_risk_error=max(max_risk_error,diff)
                        np.testing.assert_allclose(risk,a['risk'],atol=1e-12,rtol=0.)
                        np.testing.assert_array_equal(risk[0,0],risk[1,0])
                        ci=cells.index(metadata['cell'])
                        bi=metadata['block']
                        ri=tiers.index(regime)
                        si=int(sign=='minus')
                        for i,early in enumerate(labels_action):
                            for j,late in enumerate(labels_action):
                                mask=(w[0,2]==early)&(w[1,2]==late)
                                transitions[ci,bi,ri,si,i,j]=[mask.mean(),((risk[0,2]-risk[1,2])*mask).mean()]
                        for j,late in enumerate(labels_action):
                            mask=w[1,2]==late
                            chosen_terms[ci,bi,ri,si,j]=[mask.mean(),((risk[1,2]-risk[1,0])*mask).mean()]
                        np.testing.assert_allclose(chosen_terms[ci,bi,ri,si,:,1].sum(),(risk[1,2]-risk[1,0]).mean())
                        np.testing.assert_allclose(transitions[ci,bi,ri,si,:,:,1].sum(),(risk[0,2]-risk[1,2]).mean())
                        if len(a['full_ms']):
                            assert a['full_ms'].shape==(3,4) and np.all(a['full_ms']>0)
                            for order in a['timing_order']:
                                np.testing.assert_array_equal(np.sort(order),np.arange(4))
                            clusters += 1
                        count += 1
    assert len(set(seeds))==len(seeds)==512 and clusters==8 and count==192
    result=dict(blocks=len(blocks),regime_orientation_archives=count,timing_clusters=clusters,
                unique_calibration_deployment_seeds=len(set(seeds)),max_abs_risk_error=float(max_risk_error),
                method_order=list(METHODS),base_probability_range=[lower,upper],above_one_roundoff=above_one)
    from stats38 import draw, limits, samples
    idx=samples()
    def describe(arr):
        sums=arr.mean(axis=(0,1)).tolist()
        ci=limits(draw(arr,idx)).tolist()
        return dict(mean=sums,pointwise_ci=ci)
    tr=transitions.mean(axis=(2,3))
    choice=chosen_terms.mean(axis=(2,3))
    summaries=dict(action_values=labels_action.tolist(),regimes=list(tiers),
                   transition_keys=['fraction','ternary_R3_minus_R4_contribution'],
                   final_choice_keys=['fraction','ternary4_minus_fixed4_contribution'],
                   transitions=describe(tr),final_choice=describe(choice),
                   switch_fraction=describe(1.-np.trace(tr[...,0],axis1=-2,axis2=-1)),
                   by_regime={r:dict(transitions=describe(transitions[:,:,i].mean(2)),
                                     final_choice=describe(chosen_terms[:,:,i].mean(2)))
                              for i,r in enumerate(tiers)})
    (ROOT/'routing38.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (ROOT/'audit38.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
