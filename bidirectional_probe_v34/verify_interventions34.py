"""Check analytic Coulomb inversion, measured kinematics, and source action likelihoods."""
from pathlib import Path
import json
import numpy as np
from target34 import calibration,LAWS,draw_object,contact,PROBE_INCOMING,stream,sigmas,torch
from detector34 import ActionConditionedDetector
from target33 import calibration as legacy

ROOT=Path(__file__).resolve().parent

def main():
    assert not torch.cuda.is_available() or str(sigmas()[3].device)=='cpu'
    cfg=json.loads((ROOT/'policy_design34.json').read_text())
    all_errors=[];flip=0;factual=0;detector=ActionConditionedDetector();obs_max=[]
    for idx,law in enumerate(LAWS):
        alpha=cfg['alphas'][idx]
        seed=980_000_000+idx//2*10000+20
        theta=draw_object(1024,seed,10,alpha,law)
        old=legacy(seed,alpha,law);neg=calibration(seed,alpha,law,'same',-1);alt=calibration(seed,alpha,law,'alternating',-1)
        for i in range(3):np.testing.assert_array_equal(old[i],neg[i])
        np.testing.assert_array_equal(neg[0],alt[0]);np.testing.assert_array_equal(neg[1],alt[1])
        np.testing.assert_array_equal(neg[2][::2],alt[2][::2])
        if idx%2:
            near=calibration(seed,alpha,law.replace('minus','plus'),'alternating',-1)
            np.testing.assert_array_equal(near[0],alt[0])
            factual+=len(near[0])
        for schedule,first in [('same',1),('same',-1),('alternating',1),('alternating',-1)]:
            x,ids,y,d=calibration(seed,alpha,law,schedule,first)
            slope=theta.numpy()[ids,2]; mu=theta.numpy()[ids,4]; restitution=theta.numpy()[ids,5]; inp=d*abs(PROBE_INCOMING)
            n=np.stack([np.ones(8),-slope],axis=-1)/np.sqrt(1+slope[:,None]**2)
            tang=np.stack([slope,np.ones(8)],axis=-1)/np.sqrt(1+slope[:,None]**2)
            v=np.stack([np.ones(8),inp],axis=-1);vn=(v*n).sum(-1);vt=(v*tang).sum(-1)
            obj,surface,D=law.split('_');surface=cfg['surfaces'][surface]
            m=mu*(1+(1 if D=='plus' else -1)*surface['asymmetry']*np.tanh(vt/surface['speed_scale']))
            j=(1+restitution)*vn; impulse=np.clip(vt,-m*j,m*j)
            assert np.all(abs(vt)>abs(impulse))
            vout=v-j[:,None]*n-impulse[:,None]*tang
            noiseless=y-stream(seed,99).normal(size=y.shape)*sigmas()[3].cpu().numpy()
            np.testing.assert_allclose(noiseless,vout,atol=1e-12)
            all_errors.append(abs(noiseless-vout).max())
            for k in (2,4,8):
                xs=x[ids[:k]];ys=y[:k];ds=d[:k]
                combined=detector.signed_joint(xs,ys,ds)
                raw=np.concatenate([xs/detector.sigma.cpu().numpy(),ys/detector.probe_sigma.cpu().numpy()],axis=-1)
                corrected=[]
                for row,sign in zip(raw,ds):
                    b=detector.intervention_centers[int(sign)].cpu().numpy()
                    # Normalized Gaussian squared-distance density uses a different common factor.
                    z=-.5*((b-row)**2).sum(axis=-1)
                    top=z.max(axis=-1)
                    logp=top+np.log(np.exp(z-top[:,None]).mean(axis=-1))+.5*(row@row)
                    corrected.append(logp)
                err=np.max(abs(np.array(corrected)-combined));obs_max.append(err)
                np.testing.assert_allclose(corrected,combined,atol=2e-8,rtol=0)
                if np.all(ds==-1):
                    np.testing.assert_allclose(detector.joint_component_loglikelihood(xs,ys),combined,atol=1e-10,rtol=0)
            flip+=len(y)
    report=dict(heldout_root=980000000,observations_tested=flip,identical_factual_mirror_objects=factual,
        source_sobol_particles=8192,direct_independent_loglikelihood_error=max(obs_max),
        vector_impulse_vs_noiseless_target_max_abs=max(all_errors),legacy_negative_exact=True,
        incoming_magnitude=abs(PROBE_INCOMING),gpu_used=False)
    (ROOT/'action_audit34.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)
if __name__=='__main__':main()
