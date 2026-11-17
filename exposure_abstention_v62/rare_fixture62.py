"""Regenerate a prespecified pilot episode, from the original renderer's exact random stream."""
from pathlib import Path
import os
import sys
import numpy as np
import vision62 as v

def regenerate(catalog,episode):
    if not 0<=episode<v.N or not 0<=catalog<24: raise ValueError((catalog,episode))
    board_id=episode%8
    with np.load(v.ROOT/'vision_audit'/f'panel_{catalog:02}.npz') as a:
        targets=a['reset_id'][:,episode]
        truth=a['truth'][:,:,:,episode]
        electronic=a['electronic'][:,:,:,:,episode]
        expected_read=a['camera'][:,:,:,:,:,episode]
        lag=a['lag_id']
        recorded_delays=a['delays'][:,episode]
    with np.load(v.ROOT/'pilots'/f'panel_{catalog:02}.npz') as p:
        expected_code=p['codes'][:,:,:,:,:,:,episode]
    pixels=np.empty((4,2,2,2,4,2,2,8,15,27),dtype='u1')
    raw_wave=np.empty((4,2,4,2,2,2,128),dtype='u1')
    for surface in range(2):
        rng=np.random.default_rng([6207361,catalog,surface])
        schedule=np.repeat(lag[surface].reshape(-1),2)
        for chunk in range(episode//8+1):
            ii=np.arange(128*chunk,128*chunk+128)
            impact=ii%2==1
            board=(ii//16)%8
            trig,led,waves=v.instrument(rng,board,impact,schedule[ii])
            scene=v.scene(rng,128,impact,board)
        sel=ii//16==episode
        assert sel.sum()==16
        bg,y,x,amp,phase,power=[arr[sel] for arr in scene]
        bg=bg[:,:8];y=y[:,:8];x=x[:,:8]
        trig=trig[sel];led=led[sel];impact=impact[sel]
        np.testing.assert_allclose(np.stack([led,trig],-1),recorded_delays[surface].reshape(16,2),atol=2e-7)
        centers=v.TIMES[:8]+trig[:,None]
        # Same counterfactual signs as the pilot bank; no board/target label is used in the reader.
        signs=truth[:,:,surface].reshape(2,2,16)
        for exposure in (.25,.80):
            body={}
            for cause,state in np.ndindex(2,2):
                body[cause,state]=v.render(bg,y,x,amp,phase,impact,centers,exposure,signs[cause,state])
            for pulse in (.20,.80):
                d=v.CELLS.index((pulse,exposure))
                integral=np.maximum(0,np.minimum(centers+.5*exposure,led[:,None]+pulse)-
                                    np.maximum(centers-.5*exposure,led[:,None]))
                roi=bg[:,:,13:,:].copy()
                roi[:,:,:,1:5]+=(power[:,None]*integral/exposure)[:,:,None,None]
                roi=np.rint(np.clip(roi,0,255)).astype('u1')
                for cause,state in np.ndindex(2,2):
                    f=body[cause,state].copy()
                    f[:,:,13:]=roi
                    pixels[d,cause,state,surface]=f.reshape(4,2,2,8,15,27)
                raw_wave[d,surface]=waves[pulse][sel].reshape(4,2,2,2,128)
    for d,mode in np.ndindex(4,3):
        cam=v.read_camera(pixels[d],raw_wave[d],mode)
        np.testing.assert_array_equal(cam,expected_read[d,mode])
        for reset in (0,1):
            np.testing.assert_array_equal(v.encode(electronic[reset],cam,targets),expected_code[d,mode,reset])
    # At the ordinary timing episode the actual 8-bit frame and ADC buffers must also replay bitwise.
    if episode==catalog%8:
        with np.load(v.ROOT/'fixtures'/f'panel_{catalog:02}.npz') as original:
            np.testing.assert_array_equal(pixels,original['frames'])
            np.testing.assert_array_equal(raw_wave,original['wave'])
    v.save('rare_fixtures',catalog,frames=pixels,wave=raw_wave,truth=truth,electronic=electronic,
           reset_id=targets,episode=episode,board=board_id,lag_id=lag[:,episode],
           true_delays=np.broadcast_to(recorded_delays,(4,2,4,2,2,2)))
    print(catalog,episode,'code/sign OK',flush=True)

if __name__=='__main__':
    assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and os.getenv('SLURM_JOB_ID')
    for arg in sys.argv[1:]:
        c,e=map(int,arg.split(':'))
        regenerate(c,e)
