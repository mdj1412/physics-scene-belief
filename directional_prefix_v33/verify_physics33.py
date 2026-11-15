"""Independent NumPy physics, mirror pairing and practical magnitude checks on external seeds."""
from pathlib import Path
import json
import numpy as np
from target33 import LAWS, OBJECTS, SURFACES, draw_object, contact, contour, calibration, deployment_truth
from contact30 import FACTUAL_INCOMING, PROBE_INCOMING, QUERY_Y, QUERY_INCOMING, FY, VYS

ROOT=Path(__file__).resolve().parent

def main():
    cfg=json.loads((ROOT/'design_eval.json').read_text())
    results={}
    max_difference=0.
    for i,law in enumerate(LAWS):
        obj,surf,sign=law.split('_')
        geometry=OBJECTS[obj][1]; amplitude,tau=SURFACES[surf][1:]
        pair=i//2
        rows=[]
        for e,alpha in enumerate(cfg['alphas']):
            seed=840000000+pair*1000+e
            theta=draw_object(1024,seed,77,alpha,law)
            z=theta.numpy()
            _,offset,s,c,mu,r=z.T
            def boundary(y):
                y=np.asarray(y)
                return offset[:,None]+s[:,None]*y+c[:,None]*y**2/.05+geometry*y**3/(3*.025**2)
            for y,v in [(0.,FACTUAL_INCOMING),(0.,PROBE_INCOMING),(QUERY_Y,QUERY_INCOMING)]:
                eps=1e-7
                tangent=((boundary(y+eps)-boundary(y-eps))/(2*eps)).ravel()
                nx=1/np.hypot(1,tangent);ny=-tangent*nx
                normal=np.stack([nx,ny],axis=-1);tangential=np.stack([-ny,nx],axis=-1)
                vn=normal@np.array([1,v]);vt=tangential@np.array([1,v])
                coeff=mu*(1+(1 if sign=='plus' else -1)*amplitude*np.tanh(vt/tau))
                normal_delta=(1+r)*np.maximum(vn,0)
                tangential_delta=np.clip(vt,-coeff*normal_delta,coeff*normal_delta)
                velocity=[1,v]-normal*normal_delta[:,None]-tangential*tangential_delta[:,None]
                contact_mask=np.abs(vt)<=coeff*normal_delta
                truth,mask=contact(theta,y,v,law)
                np.testing.assert_array_equal(contact_mask,mask.numpy())
                np.testing.assert_allclose(velocity,truth.numpy(),atol=1.e-9,rtol=0)
                max_difference=max(max_difference,float(np.max(np.abs(velocity-truth.numpy()))))
                if v==QUERY_INCOMING:
                    rows.append([mask.float().mean().item(),np.mean(contact_mask!=(np.abs(vt)<=mu*normal_delta))])
            for y in (FY,VYS[0]):
                np.testing.assert_allclose(contour(theta,y,law).numpy(),boundary(y),atol=1e-12,rtol=0)
            if i%2==0:
                mirror=LAWS[i+1]
                a=calibration(seed,alpha,law);b=calibration(seed,alpha,mirror)
                for k in (0,1):
                    np.testing.assert_array_equal(a[k],b[k])
                label,n,obs=deployment_truth(seed,alpha,law)
                mirror_label,m,mirror_obs=deployment_truth(seed,alpha,mirror)
                np.testing.assert_array_equal(n,m)
                np.testing.assert_array_equal(obs[1],mirror_obs[1])
                assert np.any(obs[3]!=mirror_obs[3])
        results[law]=dict(checked_worlds=8*1024,heldout_audit_queries_stick_rate=float(np.mean(rows,axis=0)[0]),
                          fraction_query_changed_vs_same_object_isotropic=float(np.mean(rows,axis=0)[1]))
    out=dict(seed_base=840000000,separate_from_policy_and_prefix_panels=True, unique_audit_objects=4*8*1024,
             max_abs_velocity_error=max_difference,cells=results)
    (ROOT/'physics_audit33.json').write_text(json.dumps(out,indent=2)+'\n')
    print('checked',len(results),'cells','max velocity discrepancy',max_difference)

if __name__=='__main__':
    main()
