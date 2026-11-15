"""Evaluate a pair-specific likelihood score for weak versus local-null materials."""
import json
import numpy as np
from scipy.special import logsumexp
from gate37 import ROOT, SPEEDS, load_artifacts
from gate36 import observations, log_density, expected_gain
from routing37 import CELLS, simultaneous

def score(gates,f):
    x=f['far_x']; y=f['first3_y']; dirs=f['dirs']
    reference=gates[.50]
    ll=np.zeros((len(x),len(reference['sign'])))
    for j in range(3):
        obs=observations(x[:,j],y[:,j])
        for a,sgn in enumerate((-1,1)):
            mask=dirs[:,j]==sgn
            if mask.any():
                ll[mask]+=log_density(reference,obs[mask],a)
    rows=[]; tri=[]
    # Trailing regime indices are strong, weak, local-null, global-null
    # within each family/orientation. Condition on two possible target classes.
    for i,s in enumerate(SPEEDS):
        full=ll+log_density(gates[s],observations(x[:,3],f['fourth_y'][i]),2)
        full=full.reshape(len(x),8,4)
        le_weak=logsumexp(full[:,:,1],axis=1)
        le_null=logsumexp(full[:,:,2],axis=1)
        post=np.exp(full-logsumexp(full,axis=(1,2),keepdims=True))
        strength=np.sum(post*np.array([1.,.30,0.,0.])[None,None,:],axis=(1,2))
        np.testing.assert_allclose(strength,f['near_strength4'][i],atol=1e-12,rtol=0)
        rows.append(np.stack([f['near_strength4'][i],le_weak-le_null],axis=-1))
        weights=post.reshape(len(x),-1)
        z=f['z']
        # The gain grid is relative to the no-correction decoder, including
        # its zero-gain identity. Ties select the middle (fixed) action.
        full_value=expected_gain(gates[s],weights,z)
        np.testing.assert_allclose(full_value,f['score4'][i],atol=1e-13)
        shrink_value=expected_gain(gates[s],weights,.325*z)
        values=np.stack([shrink_value,np.zeros_like(z),full_value],axis=-1)
        selected=np.argmax(values,axis=1)
        risk=f['risk'][i]
        costs=np.stack([risk[2],risk[0],risk[1]],axis=1)
        tri_cost=np.take_along_axis(costs,selected[:,None],axis=1)[:,0]
        tri.append(np.stack([tri_cost,risk[2],risk[3],selected==0,selected==1,selected==2],axis=-1))
    return np.stack(rows),np.stack(tri)

def main():
    gates=load_artifacts()
    arr=np.zeros((32,2,2,3,8,2))
    tri_panel=np.empty((4,8,4,2,3,8,6))
    for c,cell in enumerate(CELLS):
        for b in range(8):
            folder=ROOT/'results'/cell/f'block_{b:03d}'
            for k,reg in enumerate(('strong','weak','null','global_null')):
                for oi,ori in enumerate(('plus','minus')):
                    if k==3 and oi==1:
                        tri_panel[c,b,k,oi]=tri_panel[c,b,k,0]
                        continue
                    with np.load(folder/f'{reg}_{ori}.npz') as f:
                        scores,decisions=score(gates,f)
                        tri_panel[c,b,k,oi]=decisions
                        if k in (1,2): arr[8*c+b,k-1,oi]=scores
    nrep=12000; rng=np.random.default_rng(370919)
    idx=(2*rng.integers(4,size=(nrep,4,4,2))+np.arange(2)+8*np.arange(4)[None,:,None,None]).reshape(nrep,32)
    weights=np.stack([(idx==i).sum(axis=1)/32 for i in range(32)],axis=1)
    points=np.empty((3,2)); boot=np.empty((nrep,3,2))
    for i in range(3):
        for m in range(2):
            weak=arr[:,0,:,i,:,m].reshape(32,-1)
            null=arr[:,1,:,i,:,m].reshape(32,-1)
            dif=weak[:,None,:,None]-null[None,:,None,:]
            kernel=((dif>0)+.5*(dif==0)).mean((2,3))
            points[i,m]=kernel.mean(); boot[:,i,m]=np.sum((weights@kernel)*weights,axis=1)
    contrast=points[:,1]-points[:,0]; draws=boot[:,:,1]-boot[:,:,0]
    change=points[1:,1]-points[0,1]; changes=boot[:,1:,1]-boot[:,:1,1]
    out=dict(speeds=SPEEDS,methods=['posterior_mean_strength','weak_vs_local_null_log_likelihood_ratio'],
             auroc=points.tolist(),ci95=np.quantile(boot,[.025,.975],axis=0).transpose(1,2,0).tolist(),
             method_change=contrast.tolist(),simultaneous_method_ci=simultaneous(contrast,draws).tolist(),
             specialized_speed_change=change.tolist(),simultaneous_speed_ci=simultaneous(change,changes).tolist())
    panels=tri_panel.mean(axis=(3,5))
    scopes={}
    for name,reg in [('local_equal',(0,1,2)),('weak_null',(1,2)),('strong',(0,)),('weak',(1,)),('null',(2,)),('global_null',(3,))]:
        vals=panels[:,:,reg].mean(2).reshape(32,3,6)
        point=vals.mean(0)
        contrast=np.stack((vals[...,0]-vals[...,1],vals[...,0]-vals[...,2]),axis=-1)
        e=contrast.mean(0); rep=(weights@contrast.reshape(32,6)).reshape(-1,3,2)
        speeds=e[1:]-e[:1]; srep=rep[:,1:]-rep[:,:1]
        net=point[1:,0]-point[0,0]
        vals_rep=weights@vals[...,0]
        scopes[name]=dict(means=point.tolist(),difference=e.tolist(),
                  simultaneous_ci=simultaneous(e,rep).tolist(),
                  difference_in_differences=speeds.tolist(),simultaneous_speed_ci=simultaneous(speeds,srep).tolist(),
                  absolute_speed_change=net.tolist(),absolute_speed_ci=simultaneous(net,vals_rep[:,1:]-vals_rep[:,:1]).tolist())
    tri_output=dict(scopes=scopes,speeds=SPEEDS,methods=['three_action','fixed_shrink','binary_gate'],
                    rate_order=['shrink','off','full'],contrasts=['three-fixed','three-binary'])
    (ROOT/'ternary37.json').write_text(json.dumps(tri_output,indent=2)+'\n')
    (ROOT/'binary37.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
