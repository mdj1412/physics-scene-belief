"""Test 25-mm historical equivalence and independently verify 19/31-mm bank updates on CPU."""
import json
import os
import numpy as np
from scipy.special import softmax
import runner41 as runner


def check(errors, name, a, b):
    difference=float(np.abs(np.asarray(a)-np.asarray(b)).max())
    if not np.isfinite(difference) or difference>2.e-11:
        raise ValueError((name,difference))
    errors[name]=max(difference,errors.get(name,0.))


def impulse_labels(theta,h):
    _,_,slope,bend,friction,restitution=theta.T
    tangent=slope+bend*h/.025
    return np.abs(.425+tangent)<=friction*(1+restitution)*np.maximum(0,1-tangent*.425)


def main():
    assert os.getenv('BELIEF_DEVICE')=='cpu'
    v38=runner.v38
    v38.torch.set_num_threads(1)
    bank=v38.Engine([410000301,410000302],*v38.ALPHAS[0]).inner
    theta=bank.theta.numpy()
    errors={}
    count={}
    with runner.torch.inference_mode():
        for h in (.019,.025,.031):
            locations=h+.002*np.arange(-3,4)
            curve=theta[:,1,None]+theta[:,2,None]*locations+theta[:,3,None]*locations**2/.05
            independent=curve/bank.sigma[runner.Q].numpy()
            unit=np.array([1.,-3.,.8,.2,.5,.4,-1.]);unit/=np.linalg.norm(unit)
            label=impulse_labels(theta,h)
            legacy=(bank.label[:,0].numpy()>.5)
            count[str(h)]=int((label!=legacy).sum())
            for uu in (None,unit):
                templates=runner.templates([bank],h,uu)
                white,norm,lab=[v.numpy() for v in templates[0]]
                reference=independent if uu is None else independent@(np.eye(7)-np.outer(uu,uu))
                check(errors,'reference_bank_labels',lab,label.astype(float))
                check(errors,'reference_whitened_template',white,reference)
                if h==.025 and uu is None:
                    check(errors,'legacy_whitened_template',white,bank.white[runner.Q].numpy())
                    check(errors,'legacy_candidate_label',lab,legacy.astype(float))
                ids=np.array([8,7,61,53])
                xx=bank.mu[0].numpy()[ids][None,:,:]
                acts=np.array([[runner.Q,runner.P,runner.Q,runner.P]])
                rng=np.random.default_rng(41767)
                yq=curve[ids,None,:]+rng.normal(0,.00030,(4,2,7))
                yp=bank.mu[runner.P].numpy()[ids,None,:]+rng.normal(0,.027,(4,2,2))
                ys=[{runner.Q:yq,runner.P:yp}]
                pred=runner.decode([bank],templates,xx,acts,ys)[0]
                prior=bank.fit(v38.t(xx[0])).clamp_min(1e-300).log().numpy()
                for row,action in enumerate(acts[0]):
                    if action==runner.Q:
                        mu=reference
                        y=yq[row]/bank.sigma[runner.Q].numpy()
                        if uu is not None: y=y@(np.eye(7)-np.outer(uu,uu))
                    else:
                        mu=bank.white[runner.P].numpy()
                        y=yp[row]/bank.sigma[runner.P].numpy()
                    # Direct squared-distance likelihood, compared to the production dot-product form.
                    ll=-.5*np.square(y[:,None,:]-mu[None,:,:]).sum(-1)
                    direct=softmax(prior[row,None,:]+ll,-1)@label
                    check(errors,'direct_squared_distance_prediction',direct,pred[row])
    assert count['0.025']==0 and min(count['0.019'],count['0.031'])>0
    out=dict(max_absolute_error=errors,candidate_events_changed_vs_25mm=count,candidates=len(theta))
    (runner.ROOT/'decoder_audit41.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':
    main()
