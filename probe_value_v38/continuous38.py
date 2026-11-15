"""Secondary exact maximizer of the frozen piecewise-linear source gain table."""
import json
import time
import numpy as np
import run38 as design
from gate36 import observations as features, log_density, posterior
from stats38 import samples, draw, limits


def source_optimum(artifact, x, y3, dirs, y4=None):
    ll = np.zeros((len(x),len(artifact['sign'])))
    for j in range(3):
        data=features(x[:,j],y3[:,j])
        for k,sign in enumerate((-1,1)):
            keep=dirs[:,j]==sign
            if keep.any():
                ll[keep] += log_density(artifact,data[keep],k)
    z=posterior(ll) @ artifact['far_sign']
    if y4 is not None:
        ll += log_density(artifact,features(x[:,3],y4),2)
    table=posterior(ll) @ artifact['gain']
    knots=artifact['zgrid']
    opt=np.empty_like(z)
    for i,cue in enumerate(z):
        if abs(cue)<1e-15:
            opt[i]=design.SHRINK
            continue
        weights=knots/cue
        candidates=np.r_[0., design.SHRINK, 1., weights[(weights>0)&(weights<1)]]
        gains=np.interp(candidates*cue,knots,table[i])
        winners=candidates[gains==gains.max()]
        opt[i]=winners[np.argmin(abs(winners-design.SHRINK))]
        assert gains.max()+1.e-12>=gains[:3].max()
    return z,opt


def measure(engine,artifact,records,pair,decs,xx,sensors):
    design.torch.cuda.synchronize()
    st=time.perf_counter_ns()
    selected=design.prefix_actions(engine,records,pair,xx)
    pred=np.stack([design.v33.decode_selected(d,x,a,s) for d,x,a,s in zip(decs,xx,selected,sensors)])
    x,y,d=design.observations(records)
    z,w=source_optimum(artifact,x,y[:,:3],d[:,:3],y[:,3])
    result=design.apply(pred,(z*w)[:,None],design.BETA)
    design.torch.cuda.synchronize()
    return (time.perf_counter_ns()-st)/1.e6,selected,result


def main():
    assert (design.OUT/'completion.json').exists() and design.torch.cuda.is_available()
    design.torch.set_num_threads(2)
    artifact=design.load_artifacts()[design.SPEED]
    det=design.Detector()
    loss=np.empty((4,8,3,2,2))
    rb=np.empty_like(loss)
    have_ref=True
    existing=np.empty((4,8,3,2,4))
    times=[]
    with design.torch.inference_mode():
        for c,cell in enumerate(design.CELLS):
            for b in range(8):
                dst=design.OUT/cell/f'block_{b:03d}'
                meta=json.loads((dst/'meta.json').read_text())
                oracle_file=design.ROOT/'oracle_results'/cell/dst.name/'posterior.npz'
                ref=np.load(oracle_file)['p'] if oracle_file.exists() else None
                if ref is None:
                    have_ref=False
                for s,orientation in enumerate(('plus','minus')):
                    with np.load(dst/f'shared_{orientation}.npz') as shared:
                        x,y3,d=(shared[k] for k in ('far_x','first3_y','dirs'))
                        base=shared['p_base']
                        alpha=shared['alpha']
                        for g,regime in enumerate(design.REGIMES):
                            tag=f'{regime}_{orientation}'
                            with np.load(dst/(tag+'.npz')) as arm:
                                means=[]
                                for k,y4 in enumerate((None,arm['fourth_y'])):
                                    z,w=source_optimum(artifact,x,y3,d,y4)
                                    prob=design.apply(base,(w*z)[:,None,None],design.BETA)
                                    loss[c,b,g,s,k]=((prob-arm['labels'][:,:,None])**2).mean()
                                    if ref is not None:
                                        rb[c,b,g,s,k]=((prob-ref[s,g,:,1])**2).mean()
                                    means.append(prob[:,:,0])
                                existing[c,b,g,s]=arm['risk'][1].mean(1)
                                if b>1 or tag!=design.TIMED[b][c]:
                                    continue
                                pair=meta['pair']
                                decs=[design.Engine(pair,*al) for al in alpha]
                                law=(*cell.split('_'),(1,-1)[s])
                                records=[design.world.calibration(seed,al,law,regime,meta['start'])[0]
                                         for al,seed in zip(alpha,meta['calibration_seeds'])]
                                data=[design.world.deployment(seed,al,law,regime,repeats=32)
                                      for al,seed in zip(alpha,meta['deployment_seeds'])]
                                xx=np.stack([obj[0] for obj in data])
                                sensors=[{a:obj[3][a][:,0] for a in (design.Q,design.P)} for obj in data]
                                expected3=design.apply(base[:,:,0],(z*arm['weights'][1,2])[:,None],design.BETA)
                                elapsed=np.empty((3,2)); rng=np.random.default_rng(meta['base']+3809)
                                for rep in (-1,0,1,2):
                                    for method in rng.permutation(2):
                                        if method==0:
                                            ms,a,p=design.online(det,artifact,'ternary',records,pair,decs,xx,sensors)
                                        else:
                                            ms,a,p=measure(det,artifact,records,pair,decs,xx,sensors)
                                        np.testing.assert_array_equal(a,shared['actions'])
                                        np.testing.assert_allclose(p,(expected3,means[-1])[method],rtol=0,atol=1e-8)
                                        if rep>=0:
                                            elapsed[rep,method]=ms
                                times.append(elapsed)
    idx=samples()
    d=loss.mean((2,3))
    gain=d[:,:,0]-d[:,:,1]
    diff=d[:,:,1,None]-existing.mean((2,3))
    t=np.asarray(times).reshape(4,2,3,2)
    t90=np.quantile(t,.9,axis=(0,1,2))
    ids=np.random.default_rng(381116).integers(2,size=(16000,4,2))
    bt=np.quantile(t[np.arange(4)[None,:,None],ids],.9,axis=(1,2,3))
    info=dict(loss=d.mean((0,1)),gain=gain.mean(),gain_ci=limits(draw(gain,idx)),diff=diff.mean((0,1)),
              diff_ci=limits(draw(diff,idx)),timing_ms=t90,ratio=t90[1]/t90[0],ratio_ci=limits(bt[:,1]/bt[:,0]),
              raw_timing=t,block_gain=gain)
    if have_ref:
        r=(rb[:,:,:,:,0]-rb[:,:,:,:,1]).mean((2,3))
        info.update(rb_gain=r.mean(),block_rb_gain=r)
    (design.ROOT/'continuous38.json').write_text(json.dumps(info,indent=2,default=lambda a:a.tolist())+'\n')


if __name__=='__main__':
    main()
