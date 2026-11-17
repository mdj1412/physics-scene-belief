"""Replay rare, observed R/I route changes with the original raw evidence and GPU decoder."""
import itertools
import json
import os
import time
import numpy as np
import torch
from bench62 import ROOT, METHODS, FLAGS, choice, deployment, fresh, prev, terminal, serial_alpha, select54

def run():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    records=json.loads((ROOT/'rare_selection62.json').read_text())['records']
    arms=np.array([0,1,3]); orders=list(itertools.permutations(arms))
    schedule=np.random.default_rng(6209117)
    tab=[]
    with torch.inference_mode():
        fresh.previous.install()
        engine,templates=prev.engine50()
        coef=prev.coefficients()[0]
        for j,a in enumerate(records):
            i=a['catalog']; d=a['cell']; ep=a['episode']
            with np.load(ROOT/'rare_fixtures'/f'panel_{i:02d}.npz') as f:
                assert int(f['episode'])==ep and int(f['board'])==ep%8
                causes=np.asarray(a['cause']); z=np.asarray(a['source']); mats=np.arange(2)
                frame=f['frames'][d,causes,z,mats]; wave=f['wave'][d]
                electronics=f['electronic']
                el=np.stack([electronics[:,causes[k],z[k],k] for k in mats])
                ids=f['reset_id']
            with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as f:
                codes=f['codes'][:,:,:,:,:,:,ep]
            raw,surface=deployment(i,np.asarray(a['regime']))
            obs=fresh.observe(i)
            saved_alpha=serial_alpha(raw)
            gate=select54(raw[2][:,:3],raw[3][:,:3],mode='robust')
            with np.load(ROOT/'forecasts'/f'panel_{i:02d}.npz') as f:
                plus,minus,actions=[f[k] for k in ('pplus','pminus','actions')]
            first=orders[j%6]
            block_orders=(schedule.permutation(arms),first,first[::-1])
            ms=np.zeros((2,3)); fullmask=np.zeros((3,4),bool); errs=[]
            for b,order in enumerate(block_orders):
                for arm in order:
                    col=int(np.flatnonzero(arms==arm)[0])
                    torch.cuda.synchronize(); start=time.perf_counter_ns()
                    dec,code=choice(frame,wave,el,ids,arm)
                    mixed=dec[surface]
                    need=~mixed
                    if need.any():
                        mixed[need]=select54(raw[2][need,:3],raw[3][need,:3],mode='robust')
                    alpha=terminal(raw,mixed)
                    pp,pm,act=fresh.v55.forward55(obs,engine,templates,coef)
                    prob=alpha[:,None]*pp+(1-alpha[:,None])*pm
                    torch.cuda.synchronize(); elapsed=(time.perf_counter_ns()-start)/1e6
                    if b: ms[b-1,col]=elapsed
                    expect=(a['fixed_sel'],a['fixed_v61'],[False,False])[col]
                    np.testing.assert_array_equal(dec,expect)
                    if FLAGS[arm] is not None:
                        mode,r=FLAGS[arm]
                        np.testing.assert_array_equal(code,codes[d,mode,r,causes,z,mats])
                    np.testing.assert_array_equal(mixed,dec[surface]|gate)
                    np.testing.assert_array_equal(act,actions)
                    target_alpha=saved_alpha[mixed.astype(int),np.arange(4)]
                    err=abs(prob-(target_alpha[:,None]*plus+(1-target_alpha[:,None])*minus)).max()
                    assert err<2e-9
                    errs.append(err)
                    fullmask[col]=mixed
            assert (fullmask[0]!=fullmask[1]).sum()==a['changed_sessions']
            tab.append(dict(catalog=i,episode=ep,board=ep%8,cell=d,direction=a['direction'],
                   changed_sessions=a['changed_sessions'],ms=ms.tolist(),bought=fullmask.tolist(),
                   order=[list(map(int,t)) for t in block_orders[1:]],max_error=float(max(errs))))
            print('catalog finished',i,flush=True)
    ms=np.asarray([a['ms'] for a in tab])
    direc=np.asarray([a['direction'] for a in tab])
    p=np.quantile(ms,.9,axis=(0,1))
    result=dict(methods=[METHODS[i] for i in arms],catalogs=len(tab),measured_calls_per_method=ms.shape[0]*2,
        direction_counts=np.bincount(direc,minlength=2).tolist(),gpu=torch.cuda.get_device_name(),
        p90_ms=p.tolist(),p90_differences_ms=(p[0]-p[1:]).tolist(),
        paired_median_difference_ms=np.median(ms[:,:,:1]-ms[:,:,1:],axis=(0,1)).tolist(),
        p90_by_direction_ms={str(y):np.quantile(ms[direc==y],.9,axis=(0,1)).tolist() for y in set(direc.tolist())},
        mean_changed_sessions=np.mean([x['changed_sessions'] for x in tab]),
        maximum_forecast_replay_error=max(x['max_error'] for x in tab),rows=tab)
    (ROOT/'rare_timing62.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': run()
