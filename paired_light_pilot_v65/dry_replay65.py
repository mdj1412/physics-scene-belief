"""Paired end-to-end CPU replay of archived v64 optical pilots and fresh v54 prefix calls."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
V64 = ROOT.parent / 'reflective_field_v64'
V63 = ROOT.parent / 'reflective_tracking_v63'
sys.path.insert(0,str(ROOT.parent / 'exposure_abstention_v62'))
import vision62
sys.path.insert(0,str(ROOT.parent / 'sensor_reset_v59'))
from route59 import log_odds, SOURCE_LOG_CUTOFF
sys.path.insert(0,str(ROOT.parent / 'adaptive_witness_v54'))
from policy54 import select54


def blind_route(safe_codes):
    return log_odds(safe_codes, 1, True)[0] > SOURCE_LOG_CUTOFF


def blind_raw_route(frames, wave, electronic, reset_pattern):
    signed = vision62.read_camera(frames, wave, mode=2)
    codes = vision62.encode(electronic, signed, reset_pattern)
    return codes,blind_route(codes)


def main():
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    raw_inputs = []
    den = [0,0]
    upper = [0,0]
    terminal = [0,0]
    combo_n = [0,0]
    sample = []
    prefix_count=0
    for catalog in range(24):
        name=f'panel_{catalog:02d}.npz'
        with np.load(V64/'pilots'/name) as panel:
            codes = panel['codes'][:,2,1]
        # Every field passed into either selector is observable under its arm.
        with np.load(V64/'fixtures'/name) as panel:
            frames,waves,electronic,reset,ep=(panel[k] for k in ('frames','wave','electronic','reset_id','episode'))
        assert frames.shape == (4,2,2,2,4,2,2,8,15,27)
        # References and switch/diode waveforms were captured on common simulated
        # event indices; only the rendered IMPACT pixels can vary between the two lamps.
        for width in (0,1):
            a,b=2*width,2*width+1
            np.testing.assert_array_equal(frames[a,...,0,:,:,:],frames[b,...,0,:,:,:])
            np.testing.assert_array_equal(waves[a],waves[b])
        for cell in range(4):
            rec,decision=blind_raw_route(frames[cell],waves[cell],electronic[1],reset)
            np.testing.assert_array_equal(rec,codes[cell,:,:,:,ep])
            np.testing.assert_array_equal(decision,blind_route(codes[cell,:,:,:,ep]))
            raw_inputs.append(dict(catalog=catalog,cell=cell,episode=int(ep),rows=decision.size))
        routes=blind_route(codes)
        with np.load(V63/'panels'/name) as panel:
            # Copy only the three pre-purchase observations available to both policies.
            nw=panel['nw'][...,:3]
            nf=panel['nf'][...,:3]
        gate=select54(nw,nf,mode='robust')
        assert gate.shape==(5,4,256)
        with np.load(V63/'scored63'/name) as f:
            np.testing.assert_array_equal(gate,f['gate'])
        prefix_count+=gate.size
        mat=((np.arange(4)+catalog)%4)//2
        for width in (0,1):
            left,right=routes[2*width:2*width+2]
            for material in (0,1):
                g=gate[:,mat==material].ravel()
                l=left[:,:,material].ravel()
                r=right[:,:,material].ravel()
                changed=np.logical_xor(l,r)
                assert g.size==2560 and l.size==1024
                # Cross product for each bank against every admissible prefix once.
                # The same prefix is provided on both sides; no paired trajectory labels needed.
                actual_left=l[:,None]|g[None,:]
                actual_right=r[:,None]|g[None,:]
                t=np.count_nonzero(actual_left!=actual_right)
                assert t==np.count_nonzero(changed)*np.count_nonzero(~g)
                den[width]+=len(l)
                upper[width]+=int(changed.sum())
                terminal[width]+=int(t)
                combo_n[width]+=actual_left.size
                if changed.any() and (~g).any() and len(sample)<5:
                    i=np.flatnonzero(changed)[0]; j=np.flatnonzero(~g)[0]
                    cause,hazard,episode=np.unravel_index(i,(2,2,256))
                    sample.append(dict(catalog=catalog,width=width,material=material,episode=int(episode),
                        simulation_stratum=[int(cause),int(hazard)],common_prefix_index=int(j),
                        lamp_A_impact=bool(actual_left[i,j]),lamp_B_impact=bool(actual_right[i,j])))
    ref=json.loads((V64/'sim_score64.json').read_text())['lamp_switch_uniform_width']
    check=[sum(upper)/sum(den),sum(terminal)/sum(combo_n)]
    np.testing.assert_allclose(check,[ref[0][0],ref[1][0]],rtol=0,atol=1e-12)
    out=dict(job=os.environ['SLURM_JOB_ID'],raw_code_comparisons=sum(r['rows']*4 for r in raw_inputs),
             raw_route_comparisons=sum(r['rows'] for r in raw_inputs),archived_raw_simulated_banks=48,
             distinct_simulated_calibration_banks=24*2*256,unique_calibration_impact_slots=24*2*256*8,
             replayed_v54_prefix_instances=prefix_count,upper_by_width=upper,pilot_counterfactual_rows_by_width=den,
             actual_purchases_different_by_width=terminal,prefix_pair_product_rows_by_width=combo_n,
             upper_switch=check[0],terminal_switch=check[1],five_diagnostic_traces=sample,
             run_inputs={str(f.relative_to(ROOT.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                [V64/'sim_score64.json',ROOT.parent/'adaptive_witness_v54'/'policy54.py',
                 ROOT.parent/'sensor_reset_v59'/'route59.py',ROOT.parent/'exposure_abstention_v62'/'vision62.py']})
    (ROOT/'dry_replay65.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
