"""Verify all frozen input files, paired world identifiers, sensor replay and timing."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def private_contact(theta,cell,sign,strength,speed,height):
    # At the paid/readout height and query height, the local multiplier is at
    # its plateau. This independent calculation only uses saved evaluation states.
    obj,surface=cell.split('_')
    _,_,slope,curvature,mu,e=np.moveaxis(theta,-1,0)
    h=height/.025
    slope=slope+curvature*h+(-.045*np.sin(h) if obj=='sinusoid' else .04*h**3)
    norm=np.sqrt(1+slope**2)
    vt=(speed+slope)/norm
    jn=(1+e)*np.maximum((1-slope*speed)/norm,0)
    factor=.21*2/np.pi*np.arctan(vt/.4) if surface=='atan' else .16*vt/np.sqrt(.3**2+vt**2)
    capacity=mu*(1+sign*strength*factor)*jn
    stick=np.abs(vt)<=capacity
    jt=np.clip(vt,-capacity,capacity)
    return np.stack((1-(jn+jt*slope)/norm,speed+(jn*slope-jt)/norm),-1),stick

def main():
    cfg=json.loads((ROOT/'design37.json').read_text())
    done=json.loads((DATA/'completion.json').read_text())
    hashes=done['hashes']
    for path,expected in hashes.items():
        assert digest(ROOT.parent/path)==expected,path
    clusters=0;archives=0;seeds=set();timings=[];orders=[]
    for cell_index,cell in enumerate(cfg['cells']):
        for b in range(cfg['blocks_per_cell']):
            folder=DATA/cell/f'block_{b:03d}'
            meta=json.loads((folder/'meta.json').read_text())
            assert meta['start']==(-1,1)[b%2] and meta['block']==b and meta['cell']==cell
            ids=meta['calibration_seeds']+meta['deployment_seeds']+meta['pair']
            assert len(set(ids))==18 and seeds.isdisjoint(ids)
            seeds.update(ids)
            expected_timed=cfg[f'timed_cell_block{b}'][cell_index] if b<2 else None
            assert len(meta['sha256'])==7
            same_far={}
            for filename,expected in meta['sha256'].items():
                path=folder/filename
                assert digest(path)==expected
                with np.load(path,allow_pickle=False) as d:
                    np.testing.assert_array_equal(d['speeds'],cfg['speeds'])
                    np.testing.assert_array_equal(d['methods'],cfg['methods'])
                    assert d['risk'].shape==(3,4,8)
                    assert np.isfinite(d['p_base']).all() and np.isfinite(d['risk']).all()
                    assert np.all((d['actions']==3).sum(-1)==13)
                    assert all(len(set(row))==4 for row in d['ids'])
                    np.testing.assert_array_equal(d['dirs'],np.tile(meta['start']*np.array([1,-1,1]),(8,1)))
                    reg,ori=path.stem.rsplit('_',1)
                    sign=1 if ori=='plus' else -1
                    strength={'strong':1.,'weak':.30,'null':0.,'global_null':0.}[reg]
                    for j,speed in enumerate(cfg['speeds']):
                        response,stick=private_contact(d['theta4'],cell,sign,strength,speed,cfg['height'])
                        np.testing.assert_allclose(response,d['matched_clean'][j],rtol=0,atol=5e-12)
                        np.testing.assert_array_equal(stick,d['matched_stick'][j])
                    _,label=private_contact(d['theta'],cell,sign,strength,cfg['query_speed'],cfg['query_height'])
                    np.testing.assert_array_equal(label,d['labels'])
                    assert d['accept4'].shape==(3,8)
                    np.testing.assert_array_equal(d['score4']>0,d['accept4'])
                    noise=d['fourth_y']-d['matched_clean']
                    for row in noise[1:]:np.testing.assert_allclose(row,noise[0],rtol=0,atol=1.e-12)
                    if not path.stem.startswith('global_null'):
                        side=path.stem.split('_')[-1]
                        paired=['ids','theta4','theta','far_x','first3_y','z','score3']
                        if side not in same_far:same_far[side]=[d[k] for k in paired]
                        else:
                            for k,ref in zip(paired,same_far[side]):np.testing.assert_array_equal(d[k],ref)
                    ms=d['full_ms']
                    if path.stem==expected_timed:
                        assert ms.shape==(3,3,2) and np.all(ms>0)
                        np.testing.assert_array_equal(np.sort(d['timing_order'],1),np.tile(np.arange(6),(3,1)))
                        timings.append(ms);orders.append(d['timing_order'])
                    else:assert ms.shape==(0,3,2)
                    archives+=1
            clusters+=1
    assert (clusters,archives,len(timings))==(32,224,8)
    assert done['clusters']==clusters and done['archives']==archives
    t=np.stack(timings)
    order=np.stack(orders)
    lo_ratio=[]
    for i in range(len(t)):
        per=np.quantile(np.delete(t,i,axis=0),.9,axis=(0,1))
        lo_ratio.append((per[:,1]/per[:,0]).tolist())
    per_order={}
    for j,speed in enumerate(cfg['speeds']):
        a=(np.argsort(order,axis=-1)[...,2*j] < np.argsort(order,axis=-1)[...,2*j+1])
        per_order[str(speed)]=dict(shrink_first=np.quantile(t[:,:,j][a],.9,axis=0).tolist(),
                                  gate_first=np.quantile(t[:,:,j][~a],.9,axis=0).tolist())
    out=dict(blocks=clusters,archives=archives,input_hashes=len(hashes),seeds=len(seeds),
             call_count_per_speed_method=int(t.shape[0]*t.shape[1]),timing_clusters=len(t),
             speed=cfg['speeds'],methods=cfg['timed_methods'],
             p90_ms=np.quantile(t,.9,axis=(0,1)).tolist(),repeat_p90_ms=np.quantile(t,.9,axis=0).tolist(),
             leave_one_cluster_out_gate_shrink_ratio=lo_ratio, order_p90_ms=per_order, all_checks_passed=True)
    (ROOT/'archive_audit37.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
