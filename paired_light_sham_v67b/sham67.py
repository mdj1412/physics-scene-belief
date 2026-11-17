"""Paired raw optical replay; the source-fitted route and downstream prefix gate are frozen."""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
V64 = ROOT.parent / 'reflective_field_v64'
V63 = ROOT.parent / 'reflective_tracking_v63'
sys.path.insert(0, str(V64))
import illumination64 as v64
sys.path.insert(0, str(ROOT.parent / 'paired_light_pilot_v65'))
from dry_replay65 import blind_route, select54
cam = v64.cam
N = 256
CASES = ['identical', 'slot1', 'no_added', 'marker', 'delay_raw', 'both_raw',
         'delay_corrected', 'both_corrected']
METRICS = ['active_slot', 'read_changed_per_paid', 'slot1_accepted_per_paid',
           'slot2_accepted_per_paid','slot1_wrong_accepted_per_paid','slot2_wrong_accepted_per_paid',
           'encoded_bank_changed','router_changed','terminal_flip','R_to_I','I_to_R']
NOMINAL = np.random.default_rng(670410).permutation([10,12,13,15,16,18,19,21])

def shift_adc(wave, ticks):
    ix = np.maximum(0, np.arange(128)[None,:] - ticks[:,None])
    result = wave.copy()
    result[:,1] = np.take_along_axis(wave[:,1], ix, axis=-1)
    return result

def commission():
    """Separate light pulses on clean and affected training branches. No impact sign or test wave."""
    rng = np.random.default_rng(670150)
    est=[]
    for b in range(8):
        board=np.full(128,b); impact=np.ones(128,bool); lag=np.zeros(128,int)
        raw_lags=[]
        for affected in (False,True):
            _,_,waves = cam.instrument(rng,board,impact,lag)
            w=waves[.80]
            if affected:
                w=shift_adc(w,NOMINAL[board]+rng.integers(-2,3,128))
            s,ok_s=cam.prev.edge(w[:,0]); d,ok_d=cam.prev.edge(w[:,1])
            raw_lags.append(np.median((d-s)[ok_s & ok_d]))
        est.append(raw_lags[1]-raw_lags[0])
    return np.array(est)

def signed_read(features,wave,offset):
    # Same thresholds as vision62.read_features(mode=2); correction is in clock units.
    gap,visible,led=features
    flash=(led>38.)
    sw,sv=cam.prev.edge(wave[:,0]); pd,pv=cam.prev.edge(wave[:,1])
    start=np.rint(flash.argmax(-1)-(pd-sw-offset)).astype(int)
    ok=(flash.sum(-1)==1)&sv&pv&(start>=0)&(start<=2)
    ix=np.clip(start,0,2)[:,None]+[0,1,3,4]
    g=np.take_along_axis(gap,ix,axis=-1)
    ok &= np.take_along_axis(visible,ix,axis=-1).all(-1)
    delta=(g[:,2]+g[:,3]-g[:,0]-g[:,1])*.5
    sign=np.sign(delta)
    peaks=np.partition(led,-2,axis=-1)[:,-2:]
    ok &= (abs(delta)>=.75)&((g[:,2]-g[:,0])*sign>=.20)&((g[:,3]-g[:,1])*sign>=.20)
    ok &= (peaks[:,1]>=48)&(peaks[:,0]<=28)
    return np.where(ok,sign,0).astype('i1')

def process(film,noise_a,noise_b,decoy,ticks,board,impact,wave,offset):
    base=cam.read_camera(film,wave,2)
    ii=np.flatnonzero(impact)
    out=np.broadcast_to(base,(8,len(base))).copy()
    def add(x,noise):
        return np.clip(x.astype('i2')+noise,0,255).astype('u1')
    f=film[ii]
    cleanwave=wave[ii]
    feat_a=cam.optical_features(add(f,noise_a))
    out[1,ii]=cam.read_features(feat_a,cleanwave,2)
    w_delayed=shift_adc(cleanwave,ticks)
    mask=decoy[ii]
    ghost=v64.perturb(f,.96*mask,np.zeros(len(ii),bool),np.zeros((len(ii),8),int))
    for image,idx in ((f,2),(ghost,3)):
        feat=cam.optical_features(add(image,noise_b))
        out[idx,ii]=cam.read_features(feat,cleanwave,2)
        out[idx+2,ii]=cam.read_features(feat,w_delayed,2)
        out[idx+4,ii]=signed_read(feat,w_delayed,offset[board[ii]])
    return out

def read_bank(i):
    cam.prev.N=N;cam.prev.BOARDS=v64.prev.BOARDS;cam.BOARDS=v64.prev.BOARDS
    offset=commission()
    truth,elec,targets,_=cam.prev.paid_inputs(640000+i)
    reads=np.empty((4,8)+truth.shape,'i1')
    schedule=np.argsort(np.random.default_rng([640373,i]).random((2,N,8)),axis=-1)//2
    decoy_n=0
    for m in (0,1):
        rg=np.random.default_rng([6407351,i,m]);rf=np.random.default_rng([6409211,i,m])
        dark=np.random.default_rng([67058101,i,m]);timing=np.random.default_rng([67058102,i,m])
        ids=np.random.default_rng([67058103,i,m])
        surf=v64.prev.surface(i,m);tilt=rf.uniform(-18,18)
        signs=truth[:,:,m].reshape(2,2,-1)
        lag=np.repeat(schedule[m].reshape(-1),2)
        slopes=np.empty((4,8)+signs.shape,'i1')
        for j in range(0,signs.shape[-1],128):
            ix=np.arange(j,j+128);impact=ix%2==1;board=ix//16%8
            t,led,w=cam.instrument(rg,board,impact,lag[ix]);wave=w[.80]
            bg,y,x,amp,phase,power=cam.scene(rg,len(ix),impact,board)
            bg=bg[:,:8];y=y[:,:8];x=x[:,:8]
            bg[:,:,2:5,:]+=surf['shine']*np.exp(-.5*((np.arange(27)-surf['lateral'])/2.2)**2)
            center=cam.TIMES[:8]+t[:,None];exp=.25
            dt=np.maximum(0,np.minimum(center+exp/2,led[:,None]+.80)-np.maximum(center-exp/2,led[:,None]))
            bg[:,:,13:,1:5]+=(power[:,None]*dt/exp)[:,:,None,None]
            opportunity=(rf.random(len(ix))<.35)&impact;transient=(rf.random(len(ix))<.15)&impact
            facet=tilt+rf.normal(0,9,len(ix))
            shift=(rf.integers(8,size=(len(ix),1))==np.arange(8))*rf.choice([-3,3],(len(ix),1))
            target=signs[:,:,ix];first=target[0,0];changed=(target!=first).any((0,1))
            args=(bg,y,x,amp,phase,impact,center)
            body=cam.render(*args,exp,first);alt=body.copy()
            if changed.any():alt[changed]=cam.render(*(a[changed] for a in args),exp,-first[changed])
            noises=np.rint(dark.normal(0,2.5,(2,impact.sum(),8,15,27))).astype('i2')
            ticks=NOMINAL[board[impact]]+timing.integers(-2,3,impact.sum())
            decoy=(ids.random(len(ix))<.15)&impact;decoy_n+=decoy.sum()
            for d,(angle,width) in enumerate(v64.ARMS):
                alpha=.96*opportunity*np.exp(-.5*((facet-angle/2)/width)**2)
                film=v64.perturb(body,alpha,transient,shift)
                other=film.copy()
                if changed.any():other[changed]=v64.perturb(alt[changed],alpha[changed],transient[changed],shift[changed])
                a=process(film,*noises,decoy,ticks,board,impact,wave,offset)
                b=a.copy()
                if changed.any():
                    # Preserve per-event paired noise; only the impact polarity in these shots differs.
                    imp_c=impact[changed]
                    b[:,changed]=process(other[changed],*(n[changed[impact]] for n in noises),decoy[changed],
                                        ticks[changed[impact]],board[changed],imp_c,wave[changed],offset)
                slopes[d,:,:,:,ix[0]:ix[-1]+1]=np.where(target[None]==first,a[:,None,None],b[:,None,None])
        reads[:,:,:,:,m]=slopes.reshape((4,8,2,2)+truth.shape[3:])
    panel=f'panel_{i:02}.npz'
    with np.load(V64/'paired_sim'/panel) as f:
        np.testing.assert_array_equal(reads[:,0], f['camera'][:,2])
    codes=np.stack([[cam.encode(elec[1],s,targets) for s in each] for each in reads])
    with np.load(V64/'pilots'/panel) as f:np.testing.assert_array_equal(codes[:,0], f['codes'][:,2,1])
    with np.load(V63/'panels'/panel) as f:gate=select54(f['nw'][...,:3],f['nf'][...,:3],mode='robust')
    with np.load(V63/'scored63'/panel) as f:np.testing.assert_array_equal(gate,f['gate'])
    materials=((np.arange(4)+i)%4)//2
    noimpact=np.array([(~gate[:,materials==m]).mean() for m in (0,1)])
    route=blind_route(codes)
    np.savez_compressed(ROOT/'blocks'/panel,reads=reads,codes=codes,truth=truth,routes=route,
                        gate_reference=noimpact,delay_ms=offset,decoy_impact_count=decoy_n)

def summarize(i):
    panel=f'panel_{i:02}.npz'
    with np.load(ROOT/'blocks'/panel) as f:
        reads=f['reads'];codes=f['codes'];routes=f['routes'];truth=f['truth'];gw=f['gate_reference']
        decoy_n=int(f['decoy_impact_count'])
    # dims cause, regime, material, episode, pair, object, (reference, impact).
    active=truth[...,0]!=0; t=truth[...,0]!=truth[...,1]
    pairs=[];products=[]
    for w in range(2):
        for start in (0,1):
            a=2*w+start
            for other in (0,1):
                b=2*w+(start^other)
                for k in range(8):
                    first=0 if k==0 else 1
                    ra,rb=reads[a,first],reads[b,k]
                    ca,cb=codes[a,first],codes[b,k]
                    la,lb=routes[a,first],routes[b,k]
                    ok1=(ra!=0).all(-1)&active;ok2=(rb!=0).all(-1)&active
                    wrong1=ok1 & ((ra[...,0]!=ra[...,1])!=t)
                    wrong2=ok2 & ((rb[...,0]!=rb[...,1])!=t)
                    # Changes in individual accepted sign/abstention and bank-level encoded evidence.
                    diff=(ra!=rb).any(-1)&active
                    axes=(-1,-2)
                    changed=la!=lb
                    wt=gw[None,None,:,None]
                    fs=[z.mean(axes) for z in [active,diff,ok1,ok2,wrong1,wrong2]]
                    fs += [(ca!=cb).any(-1),changed,changed*wt,((~la)&lb)*wt,(la&(~lb))*wt]
                    y=np.stack(fs,-1).reshape(2,2,2,32,8,-1).mean((0,1,2,3))
                    pairs.append(y)
                    products.append(int(np.sum(changed*wt*2560).round()))
    vals=np.stack(pairs).reshape(2,2,2,8,8,-1)
    dels=np.array(products).reshape(2,2,2,8)
    np.testing.assert_allclose(vals[...,8],vals[...,9]+vals[...,10])
    assert not vals[:,:,0,0,:,8].any()
    return vals,dels,decoy_n

def report():
    cells,term,decoys=map(np.stack,zip(*(summarize(c) for c in range(24))))
    # catalog, width, order, same/different light, condition, board, metric
    rng=np.random.default_rng(6706701)
    cat=rng.multinomial(24,np.ones(24)/24,12000)/24
    boards=rng.multinomial(8,np.ones(8)/8,12000)/8
    weights=(cat[:,:,None]*boards[:,None,:]).reshape(-1,192)
    av=cells.mean((1,2)).transpose(0,3,1,2,4).reshape(192,2,8,len(METRICS))
    center=av.mean(0)
    def band(x,tail=.025):
        boot=weights@x
        return [float(x.mean()),*np.quantile(boot,[tail,1-tail]).tolist()]
    terminal=av[...,8]
    contrast={'primary_joint_corrected_two_minus_sham':band(terminal[:,1,7]-terminal[:,0,7],.0125),
              'primary_sham_delay_raw_minus_corrected':band(terminal[:,0,4]-terminal[:,0,6],.0125)}
    for name,light,k,l in [('sham_marker_minus_clean',0,3,2),('sham_delayraw_minus_clean',0,4,2),
                           ('sham_delaycorrected_minus_clean',0,6,2),('sham_bothcorrected_minus_delaycorrected',0,7,6)]:
        contrast[name]=band(terminal[:,light,k]-terminal[:,light,l])
    for k,name in enumerate(CASES):
        contrast['two_minus_sham_'+name]=band(terminal[:,1,k]-terminal[:,0,k])
    for k,d in [(5,4),(7,6)]:
        contrast['sham_factorial_interaction_'+CASES[k]]=band(terminal[:,0,k]-terminal[:,0,d]-terminal[:,0,3]+terminal[:,0,2])
    with open(V64/'sim_score64.json') as fh:prev=json.load(fh)['lamp_switch_uniform_width'][1][0]
    np.testing.assert_allclose(center[1,0,8],prev,rtol=0,atol=1.e-12)
    np.testing.assert_array_equal(center[0,1,7:],np.zeros(4))
    np.testing.assert_allclose(term.sum((0,1,2))/(24*2*2*2048*2560),center[:,:,8])
    with np.load(ROOT/'blocks'/'panel_00.npz') as f:est=f['delay_ms'].tolist()
    out=dict(metrics=METRICS,cases=CASES,arms=['sham','two_light'],point=center.tolist(),
             terminal_ci=[[band(terminal[:,l,k]) for k in range(8)] for l in range(2)],
             contrasts=contrast,nominal_ms=(NOMINAL/16).tolist(),commissioning_ms=est,
             terminal_counts=term.sum((0,1,2)).tolist(),bank_prefix_denominator_per_cell=int(24*2*2*2048*2560),
             injected_decoy_impact_slots=int(decoys.sum()),unique_impact_slots=24*2*256*8,
             captured_missing=0,catalogs=24,boards=8,slurm_job=os.getenv('SLURM_JOB_ID'),
             directions_joint_corrected=cells[:,:,:,:,7,:,8].mean((0,1,4)).tolist(),
             widths_joint_corrected=cells[:,:,:,:,7,:,8].mean((0,2,4)).tolist())
    (ROOT/'results67.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(ROOT/'aggregate67.npz',by_catalog_board=cells,terminal_counts=term,decoy_counts=decoys)
    print('completed',contrast,flush=True)

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    (ROOT/'blocks').mkdir(exist_ok=True)
    if sys.argv[1]=='report': report()
    else:
        for arg in sys.argv[1:]:
            read_bank(int(arg))
            print('panel',arg,flush=True)
