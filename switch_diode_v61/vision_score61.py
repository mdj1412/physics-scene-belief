"""Audit held-out instruments and optical/electronic observation quality by board and catalog."""
import os
from pathlib import Path
import json
import numpy as np
from vision61 import ROOT, CELLS, BOARDS, estimate, edge, read_camera, optical_features

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    # catalog, board, timing, stats. Denominators are recorded as independent clips.
    sums=np.zeros((24,8,4,10)); read=np.zeros((24,8,4,3,4)); overlap=np.zeros((24,8,4,4))
    interaction=np.zeros((24,8,4,2))
    control=np.zeros((24,4,3))
    for c in range(24):
        with np.load(ROOT/'vision_audit'/f'panel_{c:02}.npz') as v:
            times=v['timing'][... ,1,:]
            truth=v['truth']; view=v['camera']; active=truth[...,0]!=0
            mismatch=truth[...,0]!=truth[...,1]
            good=(view[...,0]!=0)&(view[...,1]!=0)
            wrong=(view[...,0]!=view[...,1])!=mismatch[None,None]
            for b in range(8):
                sl=slice(b,None,8)
                k=times[:,:,sl].reshape(4,-1,4)
                joint=np.isfinite(k[...,2:]).all(-1)
                delta=k[...,2:]-k[...,:2]
                for d in range(4):
                    subset=delta[d,joint[d]]
                    y=k[d,joint[d],2:]
                    xi=k[d,joint[d],:2]
                    sums[c,b,d]=[len(joint[d]),joint[d].sum(),*abs(subset).sum(0),*subset.sum(0),
                                   *((y>=1)==(xi>=1)).sum(0),*(subset**2).sum(0)]
                    for mode in range(3):
                        ga=good[d,mode,:,:, :,sl]
                        er=wrong[d,mode,:,:, :,sl]
                        ac=active[:,:,:,sl]
                        read[c,b,d,mode]=[ac.sum(),(ac&ga).sum(),(ac&ga&er).sum(),(~ac&ga).sum()]
                    both=good[d,1,:,:,:,sl]&good[d,2,:,:,:,sl]&active[:,:,:,sl]
                    overlap[c,b,d]=[both.sum(),(both&wrong[d,1,:,:,:,sl]).sum(),(both&wrong[d,2,:,:,:,sl]).sum(),(good[d,2,:,:,:,sl]&active[:,:,:,sl]&~good[d,1,:,:,:,sl]).sum()]
                mask=joint.all(0)
                # Effects under a common validity mask on all four counterfactuals.
                means=k[:,mask,2:].mean(1)
                interaction[c,b]=np.stack([means[2]-means[0],means[3]-means[1],means[1]-means[0],means[3]-means[2]])
        with np.load(ROOT/'fixtures'/f'panel_{c:02}.npz') as f:
            for d in range(4):
                raw=f['frames'][d];wv=f['wave'][d]
                for ch in range(2):
                    w=wv.copy();w[...,ch,:]=24
                    control[c,d,ch]=np.count_nonzero(read_camera(raw,w))
                x=raw.copy();x[...,13:,1:5]=x[...,13:,7:11]
                control[c,d,2]=np.count_nonzero(read_camera(x,wv))
    def error(s):
        return dict(n=int(s[0]),valid=int(s[1]),coverage=s[1]/s[0],mae=(s[2:4]/s[1]).tolist(),
                   bias=(s[4:6]/s[1]).tolist(),correct_halfway=(s[6:8]/s[1]).tolist(),
                   rmse=np.sqrt(s[8:10]/s[1]).tolist())
    def panel(r):
        return dict(active=int(r[0]),read=int(r[1]),wrong=int(r[2]),invalid_read=int(r[3]),
                    yield_rate=r[1]/r[0],error_rate=r[2]/r[1] if r[1] else None)
    rng=np.random.default_rng(61184271)
    ix=rng.integers(24,size=(5000,24))
    bycat=overlap.sum(1)
    totals=bycat[ix].sum(1)
    delta=(totals[:,:,2]-totals[:,:,1])/np.maximum(1,totals[:,:,0])
    reg=interaction.mean((0,1))
    reg_ci=np.quantile(interaction.mean(1)[ix].mean(1),[.025,.975],0).transpose(1,2,0)
    out=dict(board_parameters={k:v.tolist() for k,v in BOARDS.items()},cells=CELLS,
             time_fields=['LED','camera_trigger'],delay_summary=[error(x) for x in sums.sum((0,1))],
             board_delay_summary=[[error(x) for x in row] for row in sums.sum(0)],
             readout=[[panel(m) for m in d] for d in read.sum((0,1))],
             board_readout=[[[panel(m) for m in d] for d in g] for g in read.sum(0)],
             matched_read_wrong_LED_wrong_dual_extra_dual=overlap.sum((0,1)).astype(int).tolist(),
             matched_error_dual_minus_led_pointwise_ci=np.quantile(delta,[.025,.975],0).T.tolist(),
             delta_sequence=['L2-L0 at T0','L2-L0 at T2','T2-T0 at L0','T2-T0 at L2'],
             paired_estimated_effect=reg.tolist(),paired_effect_pointwise_ci=reg_ci.tolist(),
             missing_switch_diode_LED_accepted_counts=control.sum(axis=0).astype(int).tolist())
    (ROOT/'vision61.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(ROOT/'instrument61_cells.npz',timing=sums,read_counts=read,both_read=overlap,interventions=interaction)
if __name__=='__main__':main()
