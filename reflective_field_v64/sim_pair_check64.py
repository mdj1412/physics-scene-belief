"""Additional diagnostic: common-random priorities and explicit downstream purchase verification."""
import json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
LEGACY=ROOT.parent/'reflective_tracking_v63'
sys.path.insert(0,str(LEGACY))
import score63

def main():
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    blocks=np.zeros((24,8,2,2,2))
    persistent=np.zeros((2,5))
    for i in range(24):
        rng=np.random.default_rng([6408721,i])
        with np.load(ROOT/'pilots'/f'panel_{i:02}.npz') as f:
            raw,selected=(score63.split_codes(f['codes'][:,mode,1]) for mode in (1,2))
        with np.load(ROOT/'paired_sim'/f'panel_{i:02}.npz') as f:
            signs=f['truth']
            truepair=signs[...,0]!=signs[...,1]; live=signs[...,0]!=0
            camera=f['camera'][:,1]; accepted=(camera!=0).all(-1)&live[None]
            strong=(f['alpha']>.5)&(~f['events'][None,...,1])
            wrong=((camera[...,0]!=camera[...,1])!=truepair[None])&accepted
            for w in range(2):
                l,r=2*w,2*w+1
                false_l=wrong[l]&strong[l,None,None]
                false_r=wrong[r]&strong[r,None,None]
                persistent[w]+=[live.sum(),false_l.sum(),false_r.sum(),(false_l&false_r).sum(),(false_l|false_r).sum()]
        with np.load(LEGACY/'scored63'/f'panel_{i:02}.npz') as f:
            gate=f['gate']
        shape=raw.shape[1:-1]
        s=[score63.classify(v.reshape(-1,8))[:,0].reshape(shape) for v in selected]
        for w in range(2):
            l,r=2*w,2*w+1
            mask0=raw[l]>=5;mask1=raw[r]>=5
            k0=(selected[l]>=5).sum(-1);k1=(selected[r]>=5).sum(-1)
            flex=((k0>0)&(k0<mask0.sum(-1)))|((k1>0)&(k1<mask1.sum(-1)))
            exact=(s[l]!=s[r]).astype(float)
            random=exact.copy()
            a=raw[l][flex];b=raw[r][flex];ka=k0[flex];kb=k1[flex]
            # Independent random priorities across pilots and replicas, shared across lamps
            # to ensure a change cannot be caused by rerolling an otherwise identical lottery.
            def one(symbol,k,priority):
                ranking=np.argsort(np.argsort(np.where(symbol>=5,priority,np.inf),axis=-1),axis=-1)
                blank=np.where(symbol>0,1+(symbol.astype(int)-1)%4,0)
                return score63.classify(np.where(ranking<k[:,None],symbol,blank).astype('u1'))[:,0]
            disag=np.zeros(len(a))
            for rep in range(64):
                noise=rng.random(a.shape)
                disag+=one(a,ka,noise)!=one(b,kb,noise)
            random[flex]=disag/64
            for policy,dis in enumerate((exact,random)):
                for m in (0,1):
                    # For every actual deployment v54 acquisition, count how often paired
                    # decisions differ in the terminal reference vs impact, not just gate ID.
                    p=dis[:,:,m].reshape(2,2,-1,8).mean((0,1,2))
                    actual=gate[:,((np.arange(4)+i)%4)//2==m].ravel()
                    blocks[i,:,w,policy,0]+=p/2
                    blocks[i,:,w,policy,1]+=(~actual[:,None]*p[None,:]).mean(0)/2
    expected=np.load(ROOT/'sim_score64_blocks.npz')['lamp_switch'][...,:2]
    np.testing.assert_allclose(blocks[:,:,:,0,:],expected,rtol=0,atol=1e-14)
    _,_,prod=score63.make_weights(6000,6407313)
    def interval(x):
        b=prod@x.reshape(24*8,2)
        return np.stack([b[0],*np.quantile(b[1:],[.025,.975],axis=0)],-1).tolist()
    out={'common_priority_random':interval(blocks[:,:,:,1,:].mean(2)),
         'confidence_direct_check':interval(blocks[:,:,:,0,:].mean(2)),
         'replicas_of_lottery':64,'switches':['outer','deployment_purchase'],
         'persistent_narrow_broad':persistent.astype('i8').tolist(),
         'persistent_counts':['active','wrong_L_in_strong_no_transient','wrong_R_in_strong_no_transient','both','either']}
    (ROOT/'sim_pair_check64.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
