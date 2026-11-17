from pathlib import Path
import os,json
import numpy as np
import optics63 as v

ROOT=Path(__file__).resolve().parent
readers=('v61','confidence','same_count_random')
terms=('active_slots','reads','wrong_truth','reads_with_gauge','disagree_gauge','gauge_wrong_joint','wrong_when_gauge_read')
gterms=('contacts','contact_read','contact_wrong','pairs','pair_read','pair_wrong','inactive','read_inactive')
rows=np.zeros((24,8,4,3,len(terms)))
gauges=np.zeros((24,8,len(gterms)),dtype='i8')
fault=np.zeros((24,8,4),dtype='i8')
indiv=np.zeros((24,8,4,4,3,3)) # event state, d, reader, [active,read,wrong]
assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
for i in range(24):
    with np.load(ROOT/'vision_audit'/f'panel_{i:02}.npz') as a:
        truth=a['truth']; cams=a['camera']; gauge=a['validation_sign']; events=a['events']
    with np.load(ROOT/'fixtures'/f'panel_{i:02}.npz') as f:
        np.testing.assert_array_equal(v.read_strain(f['strain_wave']),gauge[:,:,:,i%8])
    for b in range(8):
        sl=slice(b,None,8)
        ref=truth[:,:,:,sl]; active=ref[...,0]!=0; m=ref[...,0]!=ref[...,1]
        g=gauge[:,:,:,sl]; gr=(g!=0).all(-1); gm=g[...,0]!=g[...,1]
        gerror=gm!=m
        gauges[i,b]=[(ref!=0).sum(),((ref!=0)&(g!=0)).sum(),((ref!=0)&(g!=0)&(ref!=g)).sum(),
            active.sum(),(active&gr).sum(),(active&gr&gerror).sum(),(~active).sum(),(~active&gr).sum()]
        ev=events[:,sl]
        state=ev[...,0].astype(int)+2*ev[...,1]
        fault[i,b]=np.bincount(state.ravel(),minlength=4)
        for d in range(4):
            s0=cams[d,1,:,:,:,sl]
            s1=cams[d,2,:,:,:,sl]
            got0=(s0!=0).all(-1)&active; got1=(s1!=0).all(-1)&active
            n=got0.sum((-1,-2),keepdims=True)
            k=got1.sum((-1,-2),keepdims=True)
            prob=np.divide(k,n,out=np.zeros_like(k,dtype=float),where=n>0)
            for r,(s,weight) in enumerate(((s0,got0),(s1,got1),(s0,got0*prob))):
                signed=s[...,0]!=s[...,1]; wrong=m!=signed; dg=gm!=signed
                joint=weight*gr
                rows[i,b,d,r]=[active.sum(),weight.sum(),(weight*wrong).sum(),joint.sum(),
                               (joint*dg).sum(),(joint*gerror).sum(),(joint*wrong).sum()]
                for u in range(4):
                    mask=(state==u)[None,None]
                    indiv[i,b,u,d,r]=[(mask*active).sum(),(weight*mask).sum(),(weight*mask*wrong).sum()]

a=rows.sum((0,1)); rg= np.random.default_rng(632190)
cw=rg.multinomial(24,np.ones(24)/24,5000); bw=rg.multinomial(8,np.ones(8)/8,5000)
rep=np.einsum('bc,bh,chdrs->bdrs',cw,bw,rows,optimize=True)
met_names=('coverage','error_given_read','gauge_agreement_error','gauge_coverage_on_reads')
def metrics(a):
    return np.stack([a[...,1]/a[...,0],a[...,2]/a[...,1],a[...,4]/a[...,3],a[...,3]/a[...,1]],-1)
mv=metrics(a); rb=metrics(rep)
pc=np.quantile(rb,[.025,.975],axis=0).transpose(1,2,3,0)
dc=rb[:,:,1]-rb[:,:,2]
pcd=np.quantile(dc,[.025,.975],axis=0).transpose(1,2,0)
f=fault.sum((0,1))
g=gauges.sum((0,1))
out=dict(optical_cells=v.CELLS,readers=readers,count_names=terms,
         raw_counts=a.tolist(),metrics=met_names,values=mv.tolist(),intervals=pc.tolist(),
         confidence_minus_random=(mv[:,1]-mv[:,2]).tolist(),confidence_minus_random_ci=pcd.tolist(),
         event_counts_order=['neither','transient_only','coherent_only','both'],event_counts=f.tolist(),
         event_correlation=np.corrcoef(np.repeat([0,1,0,1],f),np.repeat([0,0,1,1],f))[0,1],
         strain_count_names=gterms,strain_counts=g.tolist(),raw_event_specific=indiv.sum((0,1)).tolist(),
         board_parameters={k:z.tolist() for k,z in v.BOARDS.items()},
         surfaces=[[v.surface(i,m) for m in range(2)] for i in range(24)])
# Sensor evidence distinct from optics and old board identities.
board=np.array(list(v.BOARDS.values())).T
assert np.unique(board,axis=0).shape[0]==8
for old in [v.camera_model.BOARDS,v.camera_model.prev.BOARDS]:
    assert not (board[:,None]==np.array(list(old.values())).T).all(-1).any()
np.testing.assert_allclose(a[:,1,1],a[:,2,1])
np.savez_compressed(ROOT/'optical_counts63.npz',rows=rows,gauges=gauges,event_counts=fault,conditional=indiv)
(ROOT/'optical_results63.json').write_text(json.dumps(out,indent=2)+'\n')
print('optical coverage,error,gauge-disagreement by cell and reader',mv[:,:,:3],flush=True)
print('gauge totals',g,'events',f)
