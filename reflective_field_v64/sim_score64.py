"""Equal-k policies; paired lamp change; audit-only marker sign diagnostics."""
from pathlib import Path
import json, os, sys
import numpy as np
ROOT=Path(__file__).resolve().parent
LEGACY=ROOT.parent/'reflective_tracking_v63'
sys.path.insert(0,str(LEGACY))
import score63 as base
from illumination64 import ARMS

def raw_panel(i):
    with np.load(ROOT/'pilots'/f'panel_{i:02}.npz') as f:
        r=base.classify(base.split_codes(f['codes'][:,2,1]).reshape(-1,8))[:,0].reshape(4,2,2,2,256)
    with np.load(ROOT/'paired_sim'/f'panel_{i:02}.npz') as f:
        truth=f['truth'];g=f['validation_sign'];camera=f['camera'];events=f['events'];alpha=f['alpha']
    tr=truth[...,0]!=truth[...,1];active=truth[...,0]!=0
    g_pair=(g!=0).all(-1);g_m=g[...,0]!=g[...,1]
    wrong=((camera[...,0]!=camera[...,1])!=tr[None,None])
    read=(camera!=0).all(-1)&active[None,None]
    with np.load(LEGACY/'scored63'/f'panel_{i:02}.npz') as f:
        material=((np.arange(4)+i)%4)//2
        gate=np.array([f['gate'][:,material==m].mean() for m in (0,1)])
    tab=np.empty((8,2,4))
    optical=np.empty((8,4,2,8))
    gauge=np.empty((8,3))
    for b in range(8):
        idx=slice(b,None,8)
        gauge[b]=[active[:,:,:,idx].sum(),(g_pair[:,:,:,idx]&active[:,:,:,idx]).sum(),
            ((g_m!=tr)&g_pair&active)[:,:,:,idx].sum()]
        for w in range(2):
            flip=(r[2*w]!=r[2*w+1])[:,:,:,idx]
            # Marginalizing equally over the independent target C hazard leaves same mean here:
            # both policy changes and downstream gate-R probability are independent of target regime.
            changed=flip.mean((0,1,3))
            tab[b,w,0]=changed.mean();tab[b,w,1]=(changed*(1-gate)).mean()
            for direction in (0,1):
                isfixed=r[2*w][:,:,:,idx]==direction
                tab[b,w,2+direction]=(flip&isfixed).mean()
        for d in range(4):
            no_transient=~events[:,idx,...,1]
            strong=(alpha[d,:,idx]>.5)&no_transient
            for reader in (0,1):
                keep=read[d,reader+1,:,:,:,idx]
                false=wrong[d,reader+1,:,:,:,idx]&keep
                ga=g_pair[:,:,:,idx]
                cam_disagree=((camera[d,reader+1,:,:,:,idx,...,0]!=camera[d,reader+1,:,:,:,idx,...,1])!=g_m[:,:,:,idx])
                optical[b,d,reader]=[active[:,:,:,idx].sum(),keep.sum(),false.sum(),
                    (false&strong[None,None]).sum(),(strong[None,None]&active[:,:,:,idx]).sum(),
                    (ga&keep).sum(),(ga&keep&cam_disagree).sum(),(strong[None,None]&keep).sum()]
    return tab,optical,gauge

def run():
    assert os.getenv('SLURM_JOB_ID') and os.environ['CUDA_VISIBLE_DEVICES']==''
    base.ROOT=ROOT
    routes,causes,extras,_=map(np.stack,zip(*(base.load_pilot(i) for i in range(24))))
    base.ROOT=LEGACY
    fine,ultra,gates,replay,_=map(np.stack,zip(*(base.load_deployment(i) for i in range(24))))
    route=base.target_assign(routes,'independent'); cause=base.target_assign(causes,'independent')
    cat,board,prod=base.make_weights(6000,6407313)
    out=dict(cells=ARMS,methods=base.METHODS[:3],catalogs=24,object_bank='v63 fixed holdout')
    def report(r,c):
        vals=base.risks(fine,r,c,prod)
        means=vals['mean_Brier'][:,:3]
        return dict(Brier=means[0].tolist(),
          confidence_minus_random_gate=[base.contrast(means,0,k,2) for k in (1,2)])
    out['mean']=report(route.mean(2),cause.mean(2))
    out['cell']=[report(route[:,:,d],cause[:,:,d]) for d in range(4)]
    tabs,optic,gauge=map(np.stack,zip(*(raw_panel(i) for i in range(24))))
    def stat(v):
        arr=prod@v.reshape(24*8,-1)
        return np.stack([arr[0],*np.quantile(arr[1:],[.025,.975],axis=0)],-1)
    out['lamp_switch_metrics']=['upper action L xor R','actual terminal purchase L xor R','L gate to R fixed','L fixed to R gate']
    out['lamp_switch_by_width']=[stat(tabs[:,:,d,:]).tolist() for d in range(2)]
    out['lamp_switch_uniform_width']=stat(tabs.mean(2)).tolist()
    out['optical_count_metrics']=['active opportunities','accepted','wrong','wrong and strong persistent single ghost','strong single ghost active','gauge valid accepted','gauge disagreement accepted','strong single accepted']
    out['optical_counts']=optic.sum((0,1)).astype('i8').tolist()
    out['gauge_active_valid_wrong']=gauge.sum((0,1)).astype('i8').tolist()
    assert np.allclose(tabs[:,:,:,0],tabs[:,:,:,2:].sum(-1))
    assert np.all(tabs[:,:,:,1]<=tabs[:,:,:,0]+1e-12)
    assert abs(np.mean([v['Brier'] for v in out['cell']],axis=0)-out['mean']['Brier']).max()<1e-12
    np.savez_compressed(ROOT/'sim_score64_blocks.npz',lamp_switch=tabs,optical_counts=optic,gauge_counts=gauge,fine=fine,routes=routes)
    (ROOT/'sim_score64.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
