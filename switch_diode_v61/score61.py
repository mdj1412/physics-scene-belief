"""Score frozen, paid-data-only routing on the crossed board-by-catalog design."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'sensor_reset_v59'))
from route59 import log_odds, SOURCE_LOG_CUTOFF
from score59 import MIX, EVENT, cdf_boot, weights

C, B = 24, 8
ARMS = [(2, 1), (2, 0), (1, 1), (1, 0), (0, 1), (0, 0)]
METHODS = ['dual_reset', 'dual_sham', 'led_reset', 'led_sham', 'nominal_reset',
           'nominal_sham', 'gate', 'fixed', 'dual_map_reset', 'dual_map_sham']
DELAYS = [(0,0),(0,2),(2,0),(2,2)]
TARGETS = ('independent','matched','reversed')


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def load_data():
    fine, ultra, real, gate, rates, cause, hashes = [], [], [], [], [], [], []
    for i in range(C):
        source = ROOT/'pilots'/f'panel_{i:02d}.npz'
        labels = ROOT/'scored61'/source.name
        hashes.append([digest(source),digest(labels)])
        with np.load(source) as f:
            code=f['codes']
            assert code.shape==(4,3,2,2,2,2,512,4) and code.max()<169
            if 'board_id' in f:
                assert np.array_equal(f['board_id'],np.arange(512)%B) or np.array_equal(f['board_id']%B,np.arange(512)%B)
        # From episode count to catalog x board x delay x material x cause x source x method.
        def summarize(x):
            return x.reshape(4,2,2,2,-1,B).mean(-2).transpose(4,0,3,1,2)
        acts, maps, causes = [], [], []
        for reader, reset in ARMS:
            a, mechanism = log_odds(code[:,reader,reset],reset,True)
            acts.append(summarize(a>SOURCE_LOG_CUTOFF))
            maps.append(summarize(a>1e-12))
            causes.append(summarize(mechanism>1e-12))
        zero=np.zeros_like(acts[0])
        rates.append(np.stack([*acts,zero,1-zero,*maps[:2]],axis=-1))
        cause.append(np.stack(causes,axis=-1))
        material=((np.arange(4)+i)%4)//2
        with np.load(labels) as f:
            for dest,key in ((fine,'exact_cells'),(ultra,'ultra_cells')):
                dest.append(np.stack([f[key][:,:,material==s,:2].mean((1,2)) for s in (0,1)]))
            gate.append([f['gate'][:,material==s].mean() for s in (0,1)])
            ar=f['real'][...,:2]
            assert ar.shape==(2,3,2,5,4,256,2)
            real.append(np.stack([np.take(ar,np.flatnonzero(material==s),axis=4).reshape(2,3,2,-1,2) for s in (0,1)],axis=1))
    return tuple(map(np.asarray,(fine,ultra,real,gate,rates,cause,hashes)))


def target_assign(arr,relation):
    # Cause, source regime and method are the last three axes.
    if relation=='matched':
        return arr
    out=arr.copy()
    common=arr[...,1,:,:]
    out[...,1,:,:] = common.mean(-2,keepdims=True) if relation=='independent' else common[...,::-1,:]
    return out


def components(fine, route, cause):
    # C,B,F,K,Z,M after averaging trigger states or taking a single state.
    bank=np.einsum('cfea,ze->cfza',fine,MIX)
    gap=bank[...,1]-bank[...,0]
    r=route.mean(-3)  # average mechanisms; C,B,F,Z,M
    loss=(bank[:,None,:,:,0,None]+r*gap[:,None,:,:,None]).mean((2,3))
    action=(r[:,:,:,0]+1-r[:,:,:,1]).mean(2)/2
    diagnosis=(cause[:,:,:,0]+1-cause[:,:,:,1]).mean((2,3))/2
    positive=(gap<0)[:,None,:,:,None]
    tp=(r*positive).mean((2,3)); tn=((1-r)*(~positive)).mean((2,3))
    return dict(mean_Brier=loss,AB_action_BA=action,cause_BA=diagnosis,
                tp=tp,tn=tn,pos=np.broadcast_to(positive.mean((2,3)),tp.shape))


def averages(parts,product):
    res={key:(product@val.reshape(C*B,-1)) for key,val in parts.items()}
    tp,tn,p=(res.pop(k) for k in ('tp','tn','pos'))
    res['lower_loss_action_BA']=.5*(tp/p+tn/(1-p))
    return res


def contrast(matrix,a,b,n):
    difference=matrix[:,a]-matrix[:,b]
    return [difference[0],*np.quantile(difference[1:],[.025/n,1-.025/n])]


def make_weights(n,seed):
    rng=np.random.default_rng(seed)
    cat=np.vstack([np.ones(C)/C,rng.multinomial(C,np.ones(C)/C,n)/C])
    board=np.vstack([np.ones(B)/B,rng.multinomial(B,np.ones(B)/B,n)/B])
    return cat,board,(cat[:,:,None]*board[:,None]).reshape(n+1,C*B)


def paired_session_tail(sorted_real,route,cat,board,method):
    """Exact discrete quantile on an independent board × catalog product-bootstrap draw."""
    # board-averaged action probabilities at each draw x catalog x material x target
    s=np.einsum('db,cbfza->dcfza',board,route[...,method:method+1],optimize=True)[...,0]
    arm=np.stack([1-s,s],axis=-1)
    fractions=np.einsum('dcfza,ze,t,dc->dcfeta',arm,MIX,EVENT,cat,optimize=True)/4
    h=fractions.reshape(len(cat),-1)
    flat=sorted_real.reshape(-1,sorted_real.shape[-1])
    assert np.allclose(h.sum(1),1)
    values=np.unique(flat)
    left=np.zeros(len(h),int); right=np.full(len(h),len(values)-1)
    threshold=.9*flat.shape[-1]
    while np.any(left<right):
        mid=(left+right)//2
        where, inverse=np.unique(mid,return_inverse=True)
        candidates=values[where]
        counts=np.stack([np.searchsorted(row,candidates,'right') for row in flat])
        # Gather each bootstrap draw's CDF at its own candidate without quantile-grid approximation.
        below=np.einsum('ij,ji->i',h,counts[:,inverse],optimize=False)<threshold
        left=np.where(below,mid+1,left);right=np.where(below,right,mid)
    return values[left]


def evaluate(fine,route,cause,raw,product,adjust,full):
    parts=components(fine,route,cause)
    means=averages(parts,product)
    r=route.mean(3) # C,B,F,Z,M
    dist=weights(r.mean(1))
    result={key:val[0] for key,val in means.items()}
    result['session_p90']=cdf_boot(raw,dist,np.full((1,C),1/C))[0]
    result['primary_vs_sham_gate']={}
    for key,values in means.items():
        n=1 if key=='cause_BA' else adjust
        result['primary_vs_sham_gate'][key]=[
            contrast(values,0,1,n),None if key=='cause_BA' else contrast(values,0,6,n)]
    result['dual_minus_led_reset_sham']={
        k:[contrast(v,0,2,2),contrast(v,1,3,2)] for k,v in means.items()}
    result['mean_fixed_rate_by_AB']=r.mean((0,1,2))
    if full:
        w_board=np.tile(np.eye(B)[:,None,:]/C,(1,C,1)).reshape(B,C*B)
        result['by_board']=averages(parts,w_board)
        leave=(1-np.eye(B))/(B-1)
        w_leave=np.tile(leave[:,None,:]/C,(1,C,1)).reshape(B,C*B)
        result['leave_board_out']=averages(parts,w_leave)
        result['fixed_by_cause_regime']=route.mean((0,1,2))
    return result


def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    for line in (ROOT/'score_freeze61.sha256').read_text().splitlines():
        sha,name=line.split(maxsplit=1)
        assert digest(ROOT/name)==sha
    fine,ultra,real,gate,rates,cause,hashes=load_data()
    print('loaded',rates.shape,flush=True)
    raw=np.sort(real.transpose(1,0,2,3,4,6,5),axis=-1)
    replay=np.einsum('csfeda,d->scfea',real.mean(-2),EVENT)
    cat,board,product=make_weights(6000,6104358)
    report=dict(methods=METHODS,readers=['nominal','led_only','dual'],delay_ms=DELAYS,
                n_boards=B,n_catalogs=C,n_pilot_per_board_cell=64,uniform={},cells={})
    for target in TARGETS:
        a=target_assign(rates,target)
        k=target_assign(cause,target)
        report['cells'][target]=[evaluate(fine,a[:,:,d],k[:,:,d],raw[0],product,24,False) for d in range(4)]
        v,co=a.mean(2),k.mean(2)
        primary=target=='independent'
        pack=evaluate(fine,v,co,raw[0],product,2,primary)
        r=v.mean(3)
        pack['deployment_final_impact_rate']=(gate[:,None,:,None,None]+(1-gate[:,None,:,None,None])*r).mean((0,1,2,3))
        if primary:
            tail=np.stack([paired_session_tail(raw[0],r,cat,board,i) for i in [0,1,6]],axis=-1)
            assert np.max(np.abs(tail[0]-pack['session_p90'][[0,1,6]]))<1e-8
            pack['primary_vs_sham_gate']['session_p90']=[contrast(tail,0,j,2) for j in [1,2]]
            dist=weights(r.mean(1))
            pack['seed2_p90']=cdf_boot(raw[1],dist,np.full((1,C),1/C))[0]
            for j in [0,1,6]:
                q=tail[0,[0,1,6].index(j)]
                wt=dist[j][...,None]/(C*raw.shape[-1])
                assert (wt*(raw[0]<q)).sum()<=.9+1e-10 and (wt*(raw[0]<=q)).sum()>=.9-1e-10
            pack['robustness']={}
            for name,y in zip(('ultra','binary1','binary2'),[ultra,*replay]):
                means=averages(components(y,v,co),product)['mean_Brier']
                pack['robustness'][name]=[contrast(means,0,j,2) for j in (1,6)]
        report['uniform'][target]=pack
        print('scored',target,flush=True)
    bank=np.einsum('cfea,ze->cfza',fine,MIX)
    gap=bank[...,0]-bank[...,1]
    report['gap_gate_minus_fixed_by_AB']=gap.mean((0,1))
    report['oracle_surface_loss']=bank.min(-1).mean()
    report['n_A_fixed_B_gate']=[np.sum(gap[...,0]>0),np.sum(gap[...,1]<0)]
    np.savez_compressed(ROOT/'score61_cells.npz',rates=rates,cause=cause,fine=fine,ultra=ultra,replay=replay,
                        input_hashes=hashes,gate=gate,methods=METHODS)
    (ROOT/'score61.json').write_text(json.dumps(report,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')

if __name__=='__main__':
    main()
